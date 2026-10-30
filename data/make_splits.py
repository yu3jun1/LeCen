"""固定划分。

检测：训练 subset0–6，验证 subset7，测试 subset8–9；用 pylidc 的 patient_id 检查跨划分患者，
把这类患者的全部扫描移入训练集并记录。同时按 MONAI 教程格式写出 D0a 的数据列表。
临床：合并各队列的合格患者（clinical_cohort）；clinical.development 队列 60/20/20，按 (event, treatment) 分层，
种子 2026，只使用一次；其余队列整体为外部测试（split = external）。写到 paths.clinical_meta：
clinical.csv（含 cohort、series_uid）、clinical_excluded.csv、splits_clin.csv 与 clinical_tumors.csv（合格患者的肿瘤框）。

python -m data.make_splits --cfg configs/data.yaml --part det
python -m data.make_splits --cfg configs/data.yaml --part clinical
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pylidc as pl

from common.boxes import luna_to_monai
from common.io import load_config, write_csv, write_json
from data.prep_clinical import clinical_cohort, read_tumors


def det_splits(cfg):
    folds = pd.read_csv(Path(cfg.paths.meta) / "folds.csv")
    pid = dict(pl.query(pl.Scan.series_instance_uid, pl.Scan.patient_id).all())
    by_subset = {k: s for s, ks in cfg.splits.det.items() for k in ks}
    df = folds.assign(patient_id=folds.seriesuid.map(pid), split=folds.subset.map(by_subset))
    cross = df.groupby("patient_id").split.nunique() > 1
    moved = df.patient_id.isin(cross[cross].index) & (df.split != "train")
    df["moved"] = moved
    df["note"] = np.where(moved, "patient spans splits: moved to train", "")
    df.loc[moved, "split"] = "train"
    return df[["seriesuid", "patient_id", "subset", "split", "moved", "note"]]


def monai_datalist(cfg, det):
    ann = pd.read_csv(cfg.paths.luna16_annotations)
    keys = {"train": "training", "val": "validation", "test": "test"}
    out = {v: [] for v in keys.values()}
    for r in det.itertuples():
        a = ann[ann.seriesuid == r.seriesuid]
        out[keys[r.split]].append({
            "image": f"{r.seriesuid}.nii.gz",
            "box": [luna_to_monai(c, d) for c, d in zip(a[["coordX", "coordY", "coordZ"]].to_numpy(), a.diameter_mm)],
            "label": [0] * len(a)})
    return out


def clin_splits(cfg, cohort):
    c = cfg.splits.clin
    df = cohort[["patient_id", "cohort", *c.stratify]].copy()
    df["stratum"] = df[list(c.stratify)].astype(str).agg("|".join, axis=1)
    rng = np.random.default_rng(c.seed)
    df["split"] = "external"
    for _, idx in sorted(df[df.cohort == cfg.clinical.development].groupby("stratum").groups.items()):
        idx = rng.permutation(np.array(idx))
        n_tr, n_va = round(c.ratios[0] * len(idx)), round(c.ratios[1] * len(idx))
        df.loc[idx[:n_tr], "split"] = "train"
        df.loc[idx[n_tr:n_tr + n_va], "split"] = "val"
        df.loc[idx[n_tr + n_va:], "split"] = "test"
    df["seed"] = c.seed
    return df[["patient_id", "cohort", "split", "event", "treatment", "stratum", "seed"]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--part", choices=["det", "clinical"], required=True)
    args = ap.parse_args()
    cfg = load_config(args.cfg)
    if args.part == "det":
        meta = Path(cfg.paths.meta)
        det = det_splits(cfg)
        write_csv(meta / "splits_det.csv", det)
        write_json(meta / "monai_datalist_det.json", monai_datalist(cfg, det))
        print(det.groupby("split").size().to_string(), f"\nmoved: {int(det.moved.sum())}")
    else:
        meta = Path(cfg.paths.clinical_meta)
        cohort, excluded = clinical_cohort(cfg)
        write_csv(meta / "clinical.csv", cohort)
        write_csv(meta / "clinical_excluded.csv", excluded)
        write_csv(meta / "clinical_tumors.csv", read_tumors(cfg, cohort.patient_id))
        print("临床合格：", cohort.cohort.value_counts().to_dict(), "排除：", excluded.groupby(["cohort", "reason"]).size().to_dict())
        clin = clin_splits(cfg, cohort)
        write_csv(meta / "splits_clin.csv", clin)
        print(clin.groupby(["cohort", "split", "stratum"]).size().to_string())
