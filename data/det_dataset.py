"""训练 patch：采样、增强、目标与 valid_mask。

每个样本：采样器给出 (病例, 是否正例中心, 样本种子)；正例中心为随机实例中心加 ±pos_jitter_mm 均匀扰动，
否则在体积内均匀取中心。裁出 patch，越界填 −1000 HU（标签填 0，valid_mask 填 0）。
目标：中心在 patch 内 → 正例（框保持完整尺寸）；框与 patch 相交但中心在外 → 忽略区域；
ignore.csv 中与 patch 相交的条目 → 忽略区域。忽略区域统一表示为 (中心, 半径)，单位 patch 体素。
增强（按样本种子）：三轴随机翻转（图像、标签、valid_mask、框同步）；归一化尺度上的强度缩放与平移，只作用于真实影像。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, Sampler

from common.coords import affine_of, flip_vox
from common.io import read_image, read_json, read_npz
from common.seed import generator, sample_seed

HU_MIN, HU_MAX = -1000, 400


def normalize(hu):
    return (np.clip(hu, HU_MIN, HU_MAX).astype(np.float32) - HU_MIN) / (HU_MAX - HU_MIN)


def window(start, size, shape):
    """patch [start, start+size) 与体积 [0, shape) 的重叠：返回 (体积切片, patch 切片)。"""
    lo, hi = np.maximum(start, 0), np.minimum(start + size, shape)
    return tuple(slice(l, h) for l, h in zip(lo, hi)), tuple(slice(l - s, h - s) for l, h, s in zip(lo, hi, start))


def crop(arr, start, size, fill):
    out = np.full(size, fill, arr.dtype)
    src, dst = window(start, size, arr.shape)
    out[dst] = arr[src]
    return out


def overlaps(center, half, size):
    return ((center + half >= -0.5) & (center - half <= size - 0.5)).all(-1)


def inside(center, size):
    return ((center >= -0.5) & (center <= size - 0.5)).all(-1)


class Case:
    def __init__(self, cid, vol_dir, label_dir, target_dir, token_dir):
        img = read_image(Path(vol_dir) / f"{cid}.nii.gz")
        self.id = cid
        self.vol = sitk.GetArrayFromImage(img)
        self.labels = (sitk.GetArrayFromImage(read_image(Path(label_dir) / f"{cid}.nii.gz")) if label_dir
                       else np.zeros(self.vol.shape, np.uint16))
        self.affine = affine_of(img)
        self.sp = np.array(img.GetSpacing()[::-1])
        t = read_json(Path(target_dir) / f"{cid}.json")
        self.centers = np.array([i["center_vox"] for i in t["instances"]]).reshape(-1, 3)
        self.sizes = np.array([i["size_vox"] for i in t["instances"]]).reshape(-1, 3)
        self.has_mask = np.array([i["has_mask"] for i in t["instances"]], dtype=bool)
        self.label_value = np.array([i["label_value"] for i in t["instances"]], dtype=np.int64)
        self.ign_c = np.array([i["center_vox"] for i in t["ignore"]]).reshape(-1, 3)
        self.ign_r = np.array([i["radius_vox"] for i in t["ignore"]]).reshape(-1, 3)
        tok = read_npz(Path(token_dir) / f"{cid}.npz")
        self.tokens, self.token_xyz = tok["tokens"], tok["token_world_xyz"]


class DetPatchDataset(Dataset):
    def __init__(self, cfg, ids, vol_dir, label_dir, target_dir, augment):
        self.cfg, self.ids, self.augment = cfg, list(ids), augment
        self.dirs = (vol_dir, label_dir, target_dir, Path(cfg.paths.cache) / "ctclip")
        self.patch = np.array(cfg.patch_zyx)

    def case(self, k):
        return Case(self.ids[k], *self.dirs)

    def __getitem__(self, key):
        k, pos, seed = key
        rng = np.random.default_rng(seed)
        c = self.case(k)
        if pos:
            center = c.centers[rng.integers(len(c.centers))] + rng.uniform(-1, 1, 3) * self.cfg.pos_jitter_mm / c.sp
        else:
            center = rng.uniform(0, c.vol.shape) - 0.5
        return self.patch_at(c, center, rng)

    def patch_at(self, c, center, rng):
        p = self.patch
        start = np.round(center - (p - 1) / 2).astype(int)
        image = normalize(crop(c.vol, start, p, HU_MIN))
        labels = crop(c.labels, start, p, 0).astype(np.int64)
        valid = np.zeros(p, bool)
        valid[window(start, p, c.vol.shape)[1]] = True

        ctr, half = c.centers - start, c.sizes / 2
        pos = inside(ctr, p)
        trunc = ~pos & overlaps(ctr, half, p)
        ign_c, ign_r = c.ign_c - start, c.ign_r
        keep = overlaps(ign_c, ign_r, p)
        ign = np.concatenate([np.c_[ctr[trunc], half[trunc]], np.c_[ign_c[keep], ign_r[keep]]]).reshape(-1, 6)
        boxes = np.c_[ctr[pos], c.sizes[pos]].reshape(-1, 6)

        flip = np.zeros(3)
        if self.augment:
            aug = self.cfg.augment
            for ax in aug.flip_axes:
                flip[ax] = rng.random() < 0.5
            image = image * (1 + rng.uniform(-aug.intensity.scale, aug.intensity.scale)) + rng.uniform(-aug.intensity.shift, aug.intensity.shift)
            image = np.where(valid, image, 0).astype(np.float32)
            axes = tuple(np.flatnonzero(flip))
            image, labels, valid = (np.flip(a, axes).copy() for a in (image, labels, valid))
            boxes[:, :3] = flip_vox(boxes[:, :3], flip, p)
            ign[:, :3] = flip_vox(ign[:, :3], flip, p)

        return {
            "image": torch.from_numpy(image)[None], "labels": torch.from_numpy(labels), "valid_mask": torch.from_numpy(valid),
            "boxes": torch.from_numpy(boxes).float(), "has_mask": torch.from_numpy(c.has_mask[pos]),
            "label_value": torch.from_numpy(c.label_value[pos]), "ignore": torch.from_numpy(ign).float(),
            "tokens": torch.from_numpy(c.tokens), "token_xyz": torch.from_numpy(c.token_xyz),
            "seriesuid": c.id, "offset_vox": torch.tensor(start, dtype=torch.float32),
            "flip": torch.tensor(flip, dtype=torch.float32), "spacing": torch.tensor(c.sp, dtype=torch.float32),
            "affine": torch.tensor(c.affine, dtype=torch.float32),
        }


STACK = ("image", "labels", "valid_mask", "tokens", "token_xyz", "offset_vox", "flip", "spacing", "affine")


def collate(samples):
    return {k: torch.stack([s[k] for s in samples]) if k in STACK else [s[k] for s in samples] for k in samples[0]}


def to_device(batch, dev):
    return {k: v.to(dev, non_blocking=True) if torch.is_tensor(v) else
            [x.to(dev) if torch.is_tensor(x) else x for x in v] for k, v in batch.items()}


class PatchSampler(Sampler):
    """无限采样 (病例下标, 是否正例中心, 样本种子)；使用独立 Generator，序列与 worker 数无关。"""

    def __init__(self, n_cases, pos_cases, pos_fraction, seed):
        self.n, self.pos_cases, self.pos_fraction, self.seed = n_cases, pos_cases, pos_fraction, seed

    def __iter__(self):
        g = generator(self.seed)
        epoch = 0
        while True:
            for i in range(self.n):
                pos = torch.rand(1, generator=g).item() < self.pos_fraction
                k = (self.pos_cases[torch.randint(len(self.pos_cases), (1,), generator=g).item()] if pos
                     else torch.randint(self.n, (1,), generator=g).item())
                yield k, pos, sample_seed(self.seed, epoch, i)
            epoch += 1


def luna_dataset(cfg, split, augment):
    p = Path(cfg.paths.processed)
    splits = pd.read_csv(Path(cfg.paths.meta) / "splits_det.csv")
    ids = splits[splits.split == split].seriesuid
    return DetPatchDataset(cfg, ids, p / "luna16", p / "luna16_labels", p / "luna16_targets", augment)


def loader(cfg, ds, seed):
    pos = [k for k, cid in enumerate(ds.ids) if read_json(Path(ds.dirs[2]) / f"{cid}.json")["instances"]]
    return iter(torch.utils.data.DataLoader(
        ds, batch_size=cfg.train.batch, sampler=PatchSampler(len(ds.ids), pos, cfg.pos_fraction, seed),
        num_workers=cfg.train.workers, collate_fn=collate, pin_memory=True, persistent_workers=cfg.train.workers > 0))
