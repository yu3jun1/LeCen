"""坐标与 mask 叠加图。

python -m tools.visualize --cfg configs/data.yaml --nodules 30 --out <dir>      # 随机结节：CT + 实例 mask 轮廓 + LUNA16 中心与半径
python -m tools.visualize --cfg configs/data.yaml --tokens <seriesuid> --out <dir>  # CT-CLIP token 中心叠加到预处理 CT 上
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk

from common.coords import affine_of, world_to_vox
from common.io import load_config, read_image, read_npz, world_list


def load(cfg, sid, sub):
    img = read_image(Path(cfg.paths.processed) / sub / f"{sid}.nii.gz")
    return sitk.GetArrayFromImage(img), affine_of(img), np.array(img.GetSpacing()[::-1])


def nodules(cfg, n, out):
    inst = pd.read_csv(Path(cfg.paths.meta) / "instances.csv")
    inst = inst[inst.has_mask].sample(n, random_state=0)
    for r in inst.itertuples():
        vol, a, sp = load(cfg, r.seriesuid, "luna16")
        lab = load(cfg, r.seriesuid, "luna16_labels")[0]
        c = world_to_vox(world_list([r.center_world_xyz])[0], a)[0]
        z, h = int(np.round(c[0])), 48
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(vol[z], cmap="gray", vmin=-1000, vmax=400)
        ax.contour(lab[z] == r.label_value, levels=[0.5], colors="lime", linewidths=0.8)
        ax.plot(c[2], c[1], "r+")
        ax.add_patch(plt.Circle((c[2], c[1]), r.diameter_mm / 2 / sp[2], fill=False, color="r", lw=0.8))
        ax.set_xlim(c[2] - h, c[2] + h)
        ax.set_ylim(c[1] + h, c[1] - h)
        ax.set_title(f"{r.inst_id[-24:]} {r.match_status}", fontsize=7)
        ax.axis("off")
        fig.savefig(Path(out) / f"{r.inst_id}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


def tokens(cfg, sid, out):
    vol, a, sp = load(cfg, sid, "luna16")
    t = world_to_vox(read_npz(Path(cfg.paths.cache) / "ctclip" / f"{sid}.npz")["token_world_xyz"].astype(float), a)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, k, (i, j) in zip(axes, range(3), [(1, 2), (0, 2), (0, 1)]):
        m = vol.shape[k] // 2
        ax.imshow(np.take(vol, m, axis=k), cmap="gray", vmin=-1000, vmax=400, aspect="auto")
        near = np.abs(t[:, k] - m) < 15 / sp[k]  # 池化后 token 间距为 30 mm
        ax.scatter(t[near, j], t[near, i], s=6, c="r")
        ax.set_xlim(0, vol.shape[j])
        ax.set_ylim(vol.shape[i], 0)
    fig.savefig(Path(out) / f"tokens_{sid}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--nodules", type=int)
    ap.add_argument("--tokens")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config(args.cfg)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.nodules:
        nodules(cfg, args.nodules, args.out)
    if args.tokens:
        tokens(cfg, args.tokens, args.out)
