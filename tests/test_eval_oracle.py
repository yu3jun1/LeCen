"""用参考结节中心作为预测：官方脚本与自实现 FROC 的敏感度均接近 1，CPM 一致。需要官方评测脚本与 LUNA16 标注。"""
from pathlib import Path

import pandas as pd

from common.io import ROOT, load_config
from eval.froc import froc_metrics, official, scan_records


def test_oracle(tmp_path):
    cfg = load_config(ROOT / "configs" / "data.yaml")
    splits = pd.read_csv(Path(cfg.paths.meta) / "splits_det.csv")
    sids = list(splits[splits.split == "val"].seriesuid)
    ann = pd.read_csv(cfg.paths.luna16_annotations)
    preds = ann[ann.seriesuid.isin(sids)][["seriesuid", "coordX", "coordY", "coordZ"]].assign(probability=1.0)
    preds_csv = tmp_path / "oracle.csv"
    preds.to_csv(preds_csv, index=False)
    mine = froc_metrics(scan_records(cfg, preds, sids))
    ref = official(cfg, preds_csv, sids, tmp_path / "official")
    assert mine["cpm"] > 0.99 and ref["cpm"] > 0.99
    assert abs(mine["cpm"] - ref["cpm"]) < 1e-3
