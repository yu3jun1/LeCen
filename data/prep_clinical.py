"""临床队列：合格队列与 CT 预处理。

每个队列由各自的准备脚本（data/prep_lung1.py、data/prep_acrin.py）在 clinical.cohorts.<name>.prep 下写出统一格式：
- clinical.csv：patient_id, time_days, event（1 死亡 / 0 删失 / 空 = 未知）, treatment, 以及 clinical.covariates 各列
- series.csv：patient_id, series_uid（一患者一个基线 CT 序列）
- tumors.csv：patient_id, tumor_id, bbox_world（世界坐标 cccwhd，空格分隔）, source
make_splits 合并各队列：clinical.development 指定的队列做 train/val/test 划分，其余队列整体作为外部测试（split = external）。

python -m data.prep_clinical --cfg configs/data.yaml --cohort lung1

对该队列的合格患者：读取 DICOM 序列 → 完整体积上提取 CT-CLIP token → lungmask 肺分割 → 临床裁剪规则（必要时回退）
→ 重采样保存（同 LUNA16）；肿瘤框转为 clinical_targets。肿瘤保留率只做事后审计（crop_rules --audit）。
"""
import argparse
from pathlib import Path

import pandas as pd
import SimpleITK as sitk

from common.coords import affine_of, world_to_vox
from common.io import load_config, upsert_csv, world_list, write_image, write_json
from data.cache_ctclip import cache, load_encoder
from data.crop_rules import spacing_zyx
from data.prep_luna16 import preprocess


def read_cohort(cfg, name):
    """一个队列的统一格式临床表，附 cohort 与 series_uid；clinical.csv 与 series.csv 都必须一患者一行。"""
    prep = Path(cfg.clinical.cohorts[name].prep)
    clin = pd.read_csv(prep / "clinical.csv", dtype={"patient_id": str})
    series = pd.read_csv(prep / "series.csv", dtype={"patient_id": str})
    for what, ids in (("clinical.csv", clin.patient_id), ("series.csv", series.patient_id)):
        if ids.duplicated().any():
            raise ValueError(f"{name} 的 {what} 中有重复患者：{sorted(set(ids[ids.duplicated()]))}")
    return clin.assign(cohort=name).merge(series[["patient_id", "series_uid"]], on="patient_id", how="left")


def read_tumors(cfg, patient_ids):
    """各队列 tumors.csv 中属于给定患者的肿瘤框，附 cohort。"""
    t = pd.concat([pd.read_csv(Path(c.prep) / "tumors.csv", dtype={"patient_id": str}).assign(cohort=name)
                   for name, c in cfg.clinical.cohorts.items()], ignore_index=True)
    return t[t.patient_id.isin(set(patient_ids))].reset_index(drop=True)


def clinical_cohort(cfg):
    """合并各队列的合格患者与清洗后的临床表。

    合格：event 为 0/1、time_days 为非负数、治疗已知、有基线 CT 序列；患者编号跨队列唯一。
    协变量在合并后的表上展开（数值列保留原值，类别列 one-hot），各队列得到同一组列；均附缺失指示。
    返回 (合格患者的临床表（含 cohort、series_uid）, 排除记录 patient_id/cohort/reason)。
    """
    raw = pd.concat([read_cohort(cfg, name) for name in cfg.clinical.cohorts], ignore_index=True)
    if raw.patient_id.duplicated().any():
        raise ValueError(f"不同队列间患者编号重复：{sorted(set(raw.patient_id[raw.patient_id.duplicated()]))}")
    df = raw[["patient_id", "cohort", "series_uid", "time_days", "event", "treatment"]].copy()
    for v in cfg.clinical.covariates:
        x = raw[v]
        cols = x.to_frame() if pd.api.types.is_numeric_dtype(x) else pd.get_dummies(x, prefix=v, dtype=float)
        df = pd.concat([df, cols, x.isna().astype(int).rename(f"{v}_missing")], axis=1)
    rules = [("unknown_outcome", ~df.event.isin([0, 1])),
             ("invalid_time", ~(df.time_days >= 0)),
             ("missing_treatment", df.treatment.isna()),
             ("no_baseline_ct", df.series_uid.isna())]
    reason = pd.Series("", index=df.index)
    for name, bad in rules:
        reason[(reason == "") & bad] = name
    keep = (reason == "").to_numpy()
    out = df[keep].reset_index(drop=True)
    out["event"] = out.event.astype(int)
    return out, pd.DataFrame({"patient_id": df.patient_id[~keep], "cohort": df.cohort[~keep], "reason": reason[~keep]})


def series_directory(root, series_uid):
    """优先读取 series_uid/；否则查找 patient_id/study_uid/series_uid/。"""
    root = Path(root)
    direct = root / series_uid
    if direct.is_dir():
        return direct.resolve()
    matches = {p.resolve() for p in root.glob(f"*/*/{series_uid}") if p.is_dir()}
    if not matches:
        raise FileNotFoundError(f"找不到 DICOM 序列 {series_uid}，搜索目录：{root}")
    if len(matches) != 1:
        raise ValueError(f"DICOM 序列 {series_uid} 有多个目录，需先核验：{sorted(map(str, matches))}")
    return matches.pop()


def read_series(d):
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(d)))
    return reader.Execute()


def write_targets(path, img, boxes):
    """肿瘤框（世界 cccwhd）→ 预处理体积体素坐标；临床标注没有 mask。"""
    sp = spacing_zyx(img)
    inst = [{"center_vox": world_to_vox(b[:3], affine_of(img))[0].tolist(), "size_vox": (b[[5, 4, 3]] / sp).tolist(),
             "has_mask": False, "label_value": 0} for b in boxes]
    write_json(path, {"instances": inst, "ignore": []})


def main(cfg, cohort_name):
    from lungmask import LMInferer

    p = cfg.paths
    proc, meta = Path(p.clinical_processed), Path(p.clinical_meta)
    clin = pd.read_csv(meta / "clinical.csv", dtype={"patient_id": str})
    series = clin[clin.cohort == cohort_name][["patient_id", "series_uid"]]
    tumors = pd.read_csv(meta / "clinical_tumors.csv", dtype={"patient_id": str})
    raw = cfg.clinical.cohorts[cohort_name].raw
    encoder, lm = load_encoder(cfg.ctclip.checkpoint), LMInferer()
    crops = []
    for r in series.itertuples():
        img = read_series(series_directory(raw, r.series_uid))
        cache(Path(p.cache) / "ctclip" / f"{r.patient_id}.npz", img, encoder, cfg)
        out, rec = preprocess(img, lm.apply(img), cfg.preprocess.crop.clinical, cfg)
        write_image(out, proc / "clinical" / f"{r.patient_id}.nii.gz")
        write_targets(proc / "clinical_targets" / f"{r.patient_id}.json", out,
                      world_list(tumors[tumors.patient_id == r.patient_id].bbox_world))
        crops.append({"case_id": r.patient_id, "dataset": cohort_name, **rec})
        print(r.patient_id, rec)
    upsert_csv(meta / "crops.csv", pd.DataFrame(crops), "case_id")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--cohort", required=True)
    args = ap.parse_args()
    main(load_config(args.cfg), args.cohort)
