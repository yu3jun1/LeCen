"""外扩量只由训练集计算；验证/测试的裁剪不读取任何 GT；回退只由质量规则触发。"""
import inspect
import shutil

import numpy as np
import pandas as pd
import SimpleITK as sitk
from omegaconf import OmegaConf

from data.crop_rules import base_box, fit, quality_fallback
from data.prep_luna16 import preprocess

FB = OmegaConf.create({"lung_volume_ml": [1000, 10000], "min_bbox_mm_zyx": [100, 120, 150],
                       "require_both_lungs": True, "fallback_mode": "body_bbox", "body_hu": -500})
LUNA = OmegaConf.create({"mode": "lung_bbox", "margin_mm": None, "safety_mm": 5,
                         "fit_margins_mm": [0, 5, 10, 15, 20, 25, 30], "lung_labels": [3, 4]})
SP = np.array([2.5, 1.0, 1.0])


def seg(shape=(100, 200, 200), lo=(10, 30, 20), hi=(70, 170, 180)):
    s = np.zeros(shape, np.uint8)
    s[lo[0]:hi[0], lo[1]:hi[1], lo[2]:(lo[2] + hi[2]) // 2] = 3
    s[lo[0]:hi[0], lo[1]:hi[1], (lo[2] + hi[2]) // 2:hi[2]] = 4
    return s


def test_no_gt_in_crop_signature():
    for fn in (base_box, preprocess, quality_fallback):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"ann", "annotations", "targets", "gt", "nodules", "tumors"}


def test_fallback_only_quality():
    ct = np.full((100, 200, 200), -1000, np.int16)
    ct[5:95, 10:190, 10:190] = 0
    assert quality_fallback(seg(), SP, [3, 4], FB) == ""
    one = seg()
    one[one == 4] = 0
    assert quality_fallback(one, SP, [3, 4], FB) == "missing_lung"
    assert quality_fallback(seg(lo=(10, 30, 20), hi=(12, 40, 30)), SP, [3, 4], FB) == "lung_volume"
    assert quality_fallback(seg(lo=(40, 30, 20), hi=(70, 170, 180)), SP, [3, 4], FB) == "bbox_small"
    lo, hi, mode, reason = base_box(ct, one, SP, LUNA, FB)
    assert (mode, reason) == ("body_bbox", "missing_lung")
    assert (lo == [5, 10, 10]).all() and (hi == [95, 190, 190]).all()


def write_subset(raw, k, sids, seg_dir):
    d = raw / f"subset{k}"
    d.mkdir(parents=True)
    for sid in sids:
        img = sitk.GetImageFromArray(np.zeros((100, 200, 200), np.int16))
        img.SetSpacing(tuple(SP[::-1]))
        sitk.WriteImage(img, str(d / f"{sid}.mhd"))
        s = sitk.GetImageFromArray(seg())
        s.CopyInformation(img)
        sitk.WriteImage(s, str(seg_dir / f"{sid}.mhd"))
    shutil.make_archive(str(raw / f"subset{k}"), "zip", raw, f"subset{k}")
    shutil.rmtree(d)


def test_margin_from_train_only(tmp_path):
    raw, seg_dir, meta = tmp_path / "raw", tmp_path / "seg", tmp_path / "meta"
    for d in (raw, seg_dir, meta):
        d.mkdir()
    write_subset(raw, 0, ["a"], seg_dir)
    write_subset(raw, 7, ["b"], seg_dir)
    # 训练结节越过肺框 7 mm（x 方向），验证集结节远在肺外
    pd.DataFrame({"seriesuid": ["a", "b"], "coordX": [187.0, 199.0], "coordY": [100.0, 100.0],
                  "coordZ": [100.0, 100.0], "diameter_mm": [0.0, 4.0]}).to_csv(tmp_path / "ann.csv", index=False)
    cfg = OmegaConf.create({"paths": {"raw_luna16": str(raw), "luna16_seg": str(seg_dir), "meta": str(meta),
                                      "luna16_annotations": str(tmp_path / "ann.csv")},
                            "preprocess": {"crop": {"luna": LUNA, "fallback": FB}},
                            "splits": {"det": {"train": [0], "val": [7], "test": []}}})
    assert fit(cfg) == 10 + 5
