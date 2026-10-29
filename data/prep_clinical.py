"""临床数据预处理。

python -m data.prep_clinical --cfg configs/data.yaml

合格队列由 clinical_cohort 统一确定（make_splits 写出 meta/clinical.csv 与 clinical_excluded.csv），
临床表、划分与本脚本的预处理任务都来自同一份 meta/clinical.csv。
1. idc-index 在 collection 的 CT 序列中先汇总 series_size_MB，只下载合格患者的治疗前基线 CT。
2. 读取 DICOM 序列 → 完整体积上提取 CT-CLIP token → lungmask 肺分割 → 临床裁剪规则（必要时回退）→ 重采样保存（同 LUNA16）。
3. 肿瘤保留率只做事后审计（crop_rules --audit），不据此修改任何病例的裁剪。
肿瘤框来自 meta/clinical_tumors.csv（patient_id, tumor_id, center_world_xyz, bbox_world[cccwhd], source）。
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


def clinical_cohort(cfg):
    """合格队列与清洗后的临床表。

    合格：结局可判定（原始取值属于 event_values 或 censor_values）、time_days 为非负数、治疗已知，
    且在基线 CT 清单中有对应 series。临床表与基线 CT 清单都必须是一患者一行。
    清洗：time_days 为相对统一起点的天数；event 为 0/1；数值协变量保留原值，类别协变量 one-hot；均附缺失指示。
    返回 (合格患者的临床表（含 series_uid）, 排除记录 patient_id/reason)。
    """
    c = cfg.clinical
    if not cfg.paths.clinical_series:
        raise ValueError("paths.clinical_series 尚未配置已核验的治疗前基线 CT 清单；候选表不能直接用于训练")
    raw = pd.read_csv(cfg.paths.clinical_table, dtype={c.columns.patient_id: str})
    series = pd.read_csv(cfg.paths.clinical_series, dtype={"patient_id": str})
    for name, ids in (("临床表", raw[c.columns.patient_id]), ("基线 CT 清单", series.patient_id)):
        if ids.duplicated().any():
            raise ValueError(f"{name}中有重复患者：{sorted(set(ids[ids.duplicated()]))}")
    outcome = raw[c.columns.event]
    df = pd.DataFrame({"patient_id": raw[c.columns.patient_id],
                       "time_days": pd.to_numeric(raw[c.columns.time_days], errors="coerce"),
                       "event": outcome.isin(list(c.event_values)).astype(int),
                       "treatment": raw[c.columns.treatment]})
    for v in c.covariates:
        x = raw[v]
        cols = x.to_frame() if pd.api.types.is_numeric_dtype(x) else pd.get_dummies(x, prefix=v, dtype=float)
        df = pd.concat([df, cols, x.isna().astype(int).rename(f"{v}_missing")], axis=1)
    rules = [("unknown_outcome", ~outcome.isin([*c.event_values, *c.censor_values])),
             ("invalid_time", ~(df.time_days >= 0)),
             ("missing_treatment", df.treatment.isna()),
             ("no_baseline_ct", ~df.patient_id.isin(series.patient_id))]
    reason = pd.Series("", index=df.index)
    for name, bad in rules:
        reason[(reason == "") & bad] = name
    df = df.merge(series[["patient_id", "series_uid"]], on="patient_id", how="left")
    keep = (reason == "").to_numpy()
    return df[keep].reset_index(drop=True), pd.DataFrame({"patient_id": df.patient_id[~keep], "reason": reason[~keep]})


def download(cfg, series_uids):
    from idc_index import IDCClient

    client = IDCClient()
    idx = client.index
    ct = idx[(idx.collection_id == cfg.clinical.collection) & (idx.Modality == "CT")]
    base = ct[ct.SeriesInstanceUID.isin(series_uids)]
    print(f"{cfg.clinical.collection} CT: {len(ct)} 序列 {ct.series_size_MB.sum():.0f} MB；"
          f"合格患者基线: {len(base)} 序列 {base.series_size_MB.sum():.0f} MB")
    client.download_from_selection(downloadDir=cfg.paths.raw_clinical, seriesInstanceUID=list(series_uids),
                                   dirTemplate="%SeriesInstanceUID")


def series_directory(root, series_uid):
    """优先读取 IDC 的 series_uid/；否则查找本机 TCIA 的 patient_id/study_uid/series_uid/。"""
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


def main(cfg):
    from lungmask import LMInferer

    p = cfg.paths
    proc, meta = Path(p.processed), Path(p.meta)
    series = pd.read_csv(meta / "clinical.csv", dtype={"patient_id": str})[["patient_id", "series_uid"]]
    download(cfg, series.series_uid)
    tumors = pd.read_csv(meta / "clinical_tumors.csv", dtype={"patient_id": str})
    encoder, lm = load_encoder(cfg.ctclip.checkpoint), LMInferer()
    crops = []
    for r in series.itertuples():
        img = read_series(series_directory(p.raw_clinical, r.series_uid))
        cache(Path(p.cache) / "ctclip" / f"{r.patient_id}.npz", img, encoder, cfg)
        out, rec = preprocess(img, lm.apply(img), cfg.preprocess.crop.clinical, cfg)
        write_image(out, proc / "clinical" / f"{r.patient_id}.nii.gz")
        write_targets(proc / "clinical_targets" / f"{r.patient_id}.json", out,
                      world_list(tumors[tumors.patient_id == r.patient_id].bbox_world))
        crops.append({"case_id": r.patient_id, "dataset": "clinical", **rec})
        print(r.patient_id, rec)
    upsert_csv(meta / "crops.csv", pd.DataFrame(crops), "case_id")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    main(load_config(ap.parse_args().cfg))
