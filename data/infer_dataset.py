"""全胸滑窗：窗口 = patch 尺寸，重叠 25%，不做增强；窗口越界部分填充并记录 valid_mask。

中心区域（core）：每个窗口沿各轴负责的体积区间 [lo, hi)，以相邻窗口中心的中点为界，首尾窗口延伸到体积外。
各窗口的中心区域恰好划分整个体积，每个位置只归属于中心离它最近的窗口（infer.window_core 时只保留这部分预测）。
"""
import itertools
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from common.coords import affine_of
from common.io import read_image, read_npz
from data.det_dataset import HU_MIN, crop, normalize, window


def starts(n, w, overlap):
    step = int(w * (1 - overlap))
    s = list(range(0, max(n - w, 0) + 1, step))
    if s[-1] + w < n:
        s.append(n - w)
    return s


def core(axis_starts, w):
    """{窗口起点: (lo, hi)}：相邻窗口中心的中点为界，首尾为 ±inf。"""
    c = np.array(axis_starts) + (w - 1) / 2
    mid = (c[1:] + c[:-1]) / 2
    return dict(zip(axis_starts, zip(np.r_[-np.inf, mid], np.r_[mid, np.inf])))


class InferCase:
    def __init__(self, cid, vol_dir, token_dir, patch, overlap):
        img = read_image(Path(vol_dir) / f"{cid}.nii.gz")
        self.id, self.vol, self.patch = cid, sitk.GetArrayFromImage(img), np.array(patch)
        self.shape = np.array(self.vol.shape)
        self.affine = affine_of(img)
        self.sp = np.array(img.GetSpacing()[::-1])
        tok = read_npz(Path(token_dir) / f"{cid}.npz")
        self.tokens, self.token_xyz = torch.from_numpy(tok["tokens"]), torch.from_numpy(tok["token_world_xyz"])
        axis_starts = [starts(n, w, overlap) for n, w in zip(self.shape, self.patch)]
        self.starts = [np.array(s) for s in itertools.product(*axis_starts)]
        self.core = [core(s, w) for s, w in zip(axis_starts, self.patch)]

    def batch(self, starts):
        n = len(starts)
        bounds = np.array([[self.core[a][s[a]] for a in range(3)] for s in starts])  # (n, 3, 2)
        valid = np.zeros((n, *self.patch), bool)
        for v, s in zip(valid, starts):
            v[window(s, self.patch, self.shape)[1]] = True
        return {
            "image": torch.from_numpy(np.stack([normalize(crop(self.vol, s, self.patch, HU_MIN)) for s in starts]))[:, None],
            "valid_mask": torch.from_numpy(valid),
            "core_lo": torch.tensor(bounds[..., 0], dtype=torch.float32), "core_hi": torch.tensor(bounds[..., 1], dtype=torch.float32),
            "tokens": self.tokens.expand(n, -1, -1), "token_xyz": self.token_xyz.expand(n, -1, -1),
            "offset_vox": torch.tensor(np.stack(starts), dtype=torch.float32), "flip": torch.zeros(n, 3),
            "spacing": torch.tensor(self.sp, dtype=torch.float32).expand(n, -1),
            "affine": torch.tensor(self.affine, dtype=torch.float32).expand(n, -1, -1),
        }
