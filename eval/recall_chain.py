"""信息丢失链：对每个参考结节依次判断（逐级累积）
预处理后保留 → 热图 top-N 候选覆盖（候选中心在结节半径内）→ decoder 输出命中 → NMS 后命中 → 工作点命中，按尺寸分层汇总。
工作点为该划分上 1 FP/scan 的概率阈值。保留与 crop_audit.csv 使用同一判定，只作审计。
临床队列对主肿瘤（框体积最大者）做同样统计，命中为中心落入肿瘤框外扩 cover_margin_mm，工作点为 p > p_min；
并记录回退与被截断患者比例。外部测试队列（--split external）的肿瘤框来自 BAMF-AIMI 分割，只作报告。

python -m eval.recall_chain --run runs/D1/<run_id> --split val
python -m eval.recall_chain --run runs/D1/<run_id> --split val --cohort clinical
python -m eval.recall_chain --run runs/D1/<run_id> --split external --cohort clinical
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common.coords import affine_of, world_to_vox
from common.io import ROOT, load_config, read_header, read_json, world_list, write_csv
from data.crop_rules import retained, spacing_zyx
from engine.infer_det import load_detector, predict, preds_frame, score_threshold_mask
from eval.froc import SIZE_BINS, froc, scan_records

STAGES = ["retained", "hm", "dec", "nms", "work"]


def chain(stage, kept, hit, t_work, *, inclusive=False):
    """FROC uses an inclusive threshold; clinical p_min retains strict filtering."""
    dec = hit(stage["dec_box"][:, :3])
    k = stage["keep"]
    work = score_threshold_mask(stage["dec_p"][k], t_work, inclusive=inclusive)
    raw = [kept, hit(stage["hm_xyz"]).any(), dec.any(), dec[k].any(), (dec[k] & work).any()]
    return dict(zip(STAGES, np.cumprod(raw).astype(bool)))


def work_threshold(records, target_fp=1.0):
    fps, _, thr = froc(list(records.values()))
    return thr[np.flatnonzero(fps <= target_fp)[-1]]


def luna_chain(cfg, stages, records):
    ann = pd.read_csv(cfg.paths.luna16_annotations)
    t = work_threshold(records)
    rows = []
    for sid, st in stages.items():
        h = read_header(Path(cfg.paths.processed) / "luna16" / f"{sid}.nii.gz")
        a, sp = affine_of(h), spacing_zyx(h)
        a_s = ann[ann.seriesuid == sid]
        for xyz, d in zip(a_s[["coordX", "coordY", "coordZ"]].to_numpy(float), a_s.diameter_mm):
            kept = retained(world_to_vox(xyz, a)[0], d / 2 / sp, 0, np.array(h.GetSize())[::-1])
            rows.append({"seriesuid": sid, "diameter_mm": d,
                         **chain(st, kept, lambda c: ((c - xyz) ** 2).sum(-1) < (d / 2) ** 2, t, inclusive=True)})
    return pd.DataFrame(rows)


def summarize(df):
    out = [{"group": "all", "n": len(df), **df[STAGES].mean().to_dict()}]
    for name, (lo, hi) in SIZE_BINS.items():
        g = df[(df.diameter_mm >= lo) & (df.diameter_mm < hi)]
        out.append({"group": name, "n": len(g), **g[STAGES].mean().to_dict()})
    return pd.DataFrame(out)


def chain_summary(cfg, stages, records):
    s = summarize(luna_chain(cfg, stages, records)).set_index("group")
    return {f"chain_{k}": float(s.loc["all", k]) for k in STAGES}


def clinical_chain(paths, stages, p_min, margin_mm):
    """paths：当前临床数据路径（configs/clin/base.yaml），不取检测器运行目录中的配置。"""
    meta = Path(paths.clinical_meta)
    tumors = pd.read_csv(meta / "clinical_tumors.csv", dtype={"patient_id": str})
    crops = pd.read_csv(meta / "crops.csv", dtype={"case_id": str}).set_index("case_id").fillna({"fallback_reason": ""})
    rows = []
    for pid, st in stages.items():
        t = tumors[tumors.patient_id == pid]
        if t.empty:
            continue
        box = world_list(t.bbox_world)
        box = box[np.argmax(box[:, 3:].prod(1))]
        h = read_header(Path(paths.clinical_processed) / "clinical" / f"{pid}.nii.gz")
        kept = retained(world_to_vox(box[:3], affine_of(h))[0], box[[5, 4, 3]] / 2 / spacing_zyx(h), 0, np.array(h.GetSize())[::-1])
        half = box[3:] / 2 + margin_mm
        rows.append({"patient_id": pid, "crop_mode": crops.crop_mode[pid], "fallback": crops.fallback_reason[pid] != "",
                     **chain(st, kept, lambda c: (np.abs(c - box[:3]) <= half).all(-1), p_min)})
    df = pd.DataFrame(rows)
    summary = {"n": len(df), **df[STAGES].mean().to_dict(), "fallback_ratio": df.fallback.mean(), "truncated_ratio": 1 - df.retained.mean()}
    return df, pd.DataFrame([summary])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", choices=["train", "val", "test", "external"], required=True)
    ap.add_argument("--cohort", choices=["luna16", "clinical"], default="luna16")
    args = ap.parse_args()
    run = Path(args.run)
    cfg, model = load_detector(run, "cuda")
    p_min = read_json(run / "p_min.json")["p_min"]
    if args.cohort == "luna16":
        s = pd.read_csv(Path(cfg.paths.meta) / "splits_det.csv")
        stages = predict(model, cfg, s[s.split == args.split].seriesuid, Path(cfg.paths.processed) / "luna16", "cuda")
        df = luna_chain(cfg, stages, scan_records(cfg, preds_frame(stages, p_min), list(stages)))
        summary = summarize(df)
        write_csv(run / "mech" / f"recall_chain_{args.split}_nodules.csv", df)
        write_csv(run / "mech" / "recall_chain.csv", summary)
    else:
        clin = load_config(ROOT / "configs" / "clin" / "base.yaml")
        s = pd.read_csv(Path(clin.paths.clinical_meta) / "splits_clin.csv", dtype={"patient_id": str})
        stages = predict(model, cfg, s[s.split == args.split].patient_id, Path(clin.paths.clinical_processed) / "clinical", "cuda")
        df, summary = clinical_chain(clin.paths, stages, p_min, clin.slots.cover_margin_mm)
        write_csv(run / "mech" / f"recall_chain_clinical_{args.split}_patients.csv", df)
        write_csv(run / "mech" / f"recall_chain_clinical_{args.split}.csv", summary)
    print(summary.to_string(index=False))
