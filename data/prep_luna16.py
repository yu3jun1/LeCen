"""LUNA16 逐子集预处理。

python -m data.prep_luna16 --cfg configs/data.yaml --subset 0

每个 CT：完整体积上提取 CT-CLIP token → 肺外接框 + 固定外扩裁剪（必要时按质量规则回退）→
重采样到 (1.25, 0.703125, 0.703125) mm（线性）、HU 截断到 [−1000, 400]、int16 保存 →
实例标签图（最近邻）→ 更新对应表与 targets json。子集处理完删除解压文件。
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from common.io import load_config, read_image, upsert_csv, write_image
from data.build_instances import update, write_ignore
from data.cache_ctclip import cache, load_encoder
from data.crop_rules import base_box, expand, spacing_zyx
from data.lidc_masks import make_label_map


def crop_resample(img, lo, hi, pp):
    """按体素框 [lo, hi)（zyx）裁剪，重采样到目标 spacing，HU 截断后存为 int16；几何信息写入图像头。"""
    cropped = img[int(lo[2]):int(hi[2]), int(lo[1]):int(hi[1]), int(lo[0]):int(hi[0])]
    sp = [float(s) for s in pp.spacing_zyx[::-1]]
    size = np.round(np.array(cropped.GetSize()) * np.array(cropped.GetSpacing()) / sp).astype(int)
    out = sitk.Resample(cropped, [int(s) for s in size], sitk.Transform(), sitk.sitkLinear,
                        cropped.GetOrigin(), sp, cropped.GetDirection(), pp.hu_clip[0], sitk.sitkFloat32)
    arr = np.clip(sitk.GetArrayFromImage(out), *pp.hu_clip).round().astype(np.int16)
    res = sitk.GetImageFromArray(arr)
    res.CopyInformation(out)
    return res


def preprocess(img, seg, crop, cfg):
    """返回预处理后的图像与裁剪记录；只依据 CT 与自动分割。"""
    ct = sitk.GetArrayFromImage(img)
    sp = spacing_zyx(img)
    lo, hi, mode, reason = base_box(ct, seg, sp, crop, cfg.preprocess.crop.fallback)
    lo, hi = expand(lo, hi, crop.margin_mm, sp, ct.shape)
    rec = {"crop_mode": mode, "fallback_reason": reason,
           "direction_identity": bool(np.allclose(img.GetDirection(), np.eye(3).ravel()))}
    return crop_resample(img, lo, hi, cfg.preprocess), rec


def main(cfg, subset):
    p = cfg.paths
    proc, cache_dir = Path(p.processed), Path(p.cache)
    write_ignore(cfg)
    raw = Path(p.raw_luna16)
    shutil.unpack_archive(raw / f"subset{subset}.zip", raw)
    encoder = load_encoder(cfg.ctclip.checkpoint)
    crops, folds = [], []
    for mhd in sorted((raw / f"subset{subset}").glob("*.mhd")):
        sid = mhd.stem
        img = read_image(mhd)
        cache(cache_dir / "ctclip" / f"{sid}.npz", img, encoder, cfg)
        seg = sitk.GetArrayFromImage(read_image(Path(p.luna16_seg) / mhd.name))
        out, rec = preprocess(img, seg, cfg.preprocess.crop.luna, cfg)
        write_image(out, proc / "luna16" / f"{sid}.nii.gz")
        label, clusters, patient_id = make_label_map(sid, out)
        write_image(label, proc / "luna16_labels" / f"{sid}.nii.gz")
        update(cfg, sid, label, clusters, patient_id)
        crops.append({"case_id": sid, "dataset": "luna16", **rec})
        folds.append({"seriesuid": sid, "subset": subset})
        print(sid, rec)
    upsert_csv(Path(p.meta) / "crops.csv", pd.DataFrame(crops), "case_id")
    upsert_csv(Path(p.meta) / "folds.csv", pd.DataFrame(folds), "seriesuid")
    shutil.rmtree(raw / f"subset{subset}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--subset", type=int, required=True)
    args = ap.parse_args()
    main(load_config(args.cfg), args.subset)
