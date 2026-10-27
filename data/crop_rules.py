"""裁剪规则：在训练集上确定 LUNA16 外扩量；回退只依据图像与分割质量；保留率事后审计（不回写预处理）。

python -m data.crop_rules --cfg configs/data.yaml --fit-on train   # 确定外扩量并写回配置
python -m data.crop_rules --cfg configs/data.yaml --audit          # 事后保留率审计
"""
import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

from common.coords import affine_of, world_to_vox
from common.io import load_config, read_header, read_image, world_list, write_csv


def spacing_zyx(img):
    return np.array(img.GetSpacing()[::-1])


def bbox(mask):
    idx = np.argwhere(mask)
    return idx.min(0), idx.max(0) + 1


def lung_bbox(seg, labels):
    return bbox(np.isin(seg, labels))


def body_bbox(ct, hu):
    lab, n = ndimage.label(ct > hu)
    largest = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    return bbox(lab == largest)


def quality_fallback(seg, sp, labels, fb):
    """返回回退原因；空字符串表示自动肺分割可用。"""
    vol_ml = np.isin(seg, labels).sum() * sp.prod() / 1000
    if not fb.lung_volume_ml[0] <= vol_ml <= fb.lung_volume_ml[1]:
        return "lung_volume"
    if fb.require_both_lungs and any((seg == l).sum() == 0 for l in labels):
        return "missing_lung"
    lo, hi = lung_bbox(seg, labels)
    if ((hi - lo) * sp < np.array(fb.min_bbox_mm_zyx)).any():
        return "bbox_small"
    return ""


def base_box(ct, seg, sp, crop, fb):
    """外扩前的裁剪框（体素索引 [lo, hi)）、裁剪方式与回退原因。只读 CT 与自动分割。"""
    reason = quality_fallback(seg, sp, crop.lung_labels, fb)
    mode = fb.fallback_mode if reason else crop.mode
    boxes = {"lung_bbox": lambda: lung_bbox(seg, crop.lung_labels), "body_bbox": lambda: body_bbox(ct, fb.body_hu)}
    lo, hi = boxes[mode]()
    return lo, hi, mode, reason


def expand(lo, hi, margin_mm, sp, shape):
    m = np.round(margin_mm / sp).astype(int)
    return np.maximum(lo - m, 0), np.minimum(hi + m, shape)


def retained(center_vox, half_vox, lo, hi):
    """目标（中心 ± 半径/半边长）完整落在体素范围 [lo, hi) 内。"""
    return ((center_vox - half_vox >= lo - 0.5) & (center_vox + half_vox <= hi - 0.5)).all(-1)


def luna_targets(ann, sid, a, sp):
    rows = ann[ann.seriesuid == sid]
    center = world_to_vox(rows[["coordX", "coordY", "coordZ"]].to_numpy(), a)
    return center, rows.diameter_mm.to_numpy()[:, None] / 2 / sp


def fit(cfg):
    p, crop = cfg.paths, cfg.preprocess.crop
    ann = pd.read_csv(p.luna16_annotations)
    rows = []
    raw = Path(p.raw_luna16)
    for k in cfg.splits.det.train:
        shutil.unpack_archive(raw / f"subset{k}.zip", raw)
        for mhd in sorted((raw / f"subset{k}").glob("*.mhd")):
            img = read_image(mhd)
            seg = sitk.GetArrayFromImage(read_image(Path(p.luna16_seg) / mhd.name))
            sp = spacing_zyx(img)
            lo, hi, _, _ = base_box(sitk.GetArrayFromImage(img), seg, sp, crop.luna, crop.fallback)
            c, r = luna_targets(ann, mhd.stem, affine_of(img), sp)
            for m in crop.luna.fit_margins_mm:
                lo_m, hi_m = expand(lo, hi, m, sp, seg.shape)
                rows.append({"margin_mm": m, "n_targets": len(c), "n_retained": int(retained(c, r, lo_m, hi_m).sum())})
        shutil.rmtree(raw / f"subset{k}")
    tab = pd.DataFrame(rows).groupby("margin_mm", as_index=False).sum()
    tab["retention"] = tab.n_retained / tab.n_targets
    write_csv(Path(p.meta) / "crop_fit.csv", tab)
    margin = float(tab[tab.retention == 1.0].margin_mm.min() + crop.luna.safety_mm)
    print(tab.to_string(index=False), f"\nmargin_mm = {margin}")
    return margin


def write_margin(cfg_path, margin):
    text = Path(cfg_path).read_text()
    text = re.sub(r"(luna:\n\s+mode: \S+\s*(?:#[^\n]*)?\n\s+margin_mm: )\S+", rf"\g<1>{margin:g}", text)
    Path(cfg_path).write_text(text)


def audit(cfg):
    p = cfg.paths
    meta = Path(p.meta)
    crops = pd.read_csv(meta / "crops.csv", dtype={"case_id": str}).set_index("case_id")
    rows = []
    ann = pd.read_csv(p.luna16_annotations)
    for r in pd.read_csv(meta / "splits_det.csv").itertuples():
        h = read_header(Path(p.processed) / "luna16" / f"{r.seriesuid}.nii.gz")
        sp = spacing_zyx(h)
        c, half = luna_targets(ann, r.seriesuid, affine_of(h), sp)
        rows.append((r.seriesuid, r.split, c, half, h))
    tumors = pd.read_csv(meta / "clinical_tumors.csv", dtype={"patient_id": str})
    for r in pd.read_csv(meta / "splits_clin.csv", dtype={"patient_id": str}).itertuples():
        h = read_header(Path(p.processed) / "clinical" / f"{r.patient_id}.nii.gz")
        box = world_list(tumors[tumors.patient_id == r.patient_id].bbox_world)
        c = world_to_vox(box[:, :3], affine_of(h))
        rows.append((r.patient_id, r.split, c, box[:, [5, 4, 3]] / 2 / spacing_zyx(h), h))
    out = []
    for case, split, c, half, h in rows:
        ok = retained(c, half, 0, np.array(h.GetSize())[::-1])
        out.append({"case_id": case, "split": split, "crop_mode": crops.crop_mode[case],
                    "fallback_reason": crops.fallback_reason[case], "n_targets": len(ok), "n_retained": int(ok.sum())})
    df = pd.DataFrame(out).fillna({"fallback_reason": ""})
    write_csv(meta / "crop_audit.csv", df)
    s = df.groupby("split").agg(n_targets=("n_targets", "sum"), n_retained=("n_retained", "sum"),
                                n_fallback=("fallback_reason", lambda x: (x != "").sum()))
    s["retention"] = s.n_retained / s.n_targets
    print(s.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--fit-on", choices=["train"])
    ap.add_argument("--audit", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.cfg)
    if args.fit_on:
        write_margin(args.cfg, fit(cfg))
    if args.audit:
        audit(cfg)
