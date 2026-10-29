"""检测评测：官方脚本 + 自实现 FROC（严格复现官方命中规则），含按尺寸敏感度与按患者 bootstrap。

命中规则：候选中心落在参考结节半径内为命中（同一结节多个命中只取最大概率计一次；一个候选可命中多个结节）；
落在 irrelevant findings（ignore.csv，直径无效时为 10 mm）内且未命中参考结节的候选既不算 TP 也不算 FP。
按尺寸敏感度：先用全部候选确定达到目标 FP/scan 的概率阈值，再在该阈值下分别统计各尺寸组。

python -m eval.froc --run runs/D1/<run_id> --split test
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from common.io import ROOT, load_config, world_list, write_csv, write_json

FP_POINTS = [0.125, 0.25, 0.5, 1, 2, 4, 8]
SIZE_BINS = {"3to6": (3, 6), "6to10": (6, 10), "10plus": (10, np.inf)}
# Existing run snapshots may predate this path field; use the local data config
# as their default, while allowing a run to pin its own evaluation script.
OFFICIAL = Path(load_config(ROOT / "configs" / "data.yaml").paths.luna16_evaluation_script)
XYZ = ["coordX", "coordY", "coordZ"]


def scan_records(cfg, preds, sids):
    """每个 scan：各参考结节的最大命中概率（未命中为 −inf）与直径，以及 FP 概率。"""
    ann = pd.read_csv(cfg.paths.luna16_annotations)
    ign = pd.read_csv(Path(cfg.paths.meta) / "ignore.csv")
    rec = {}
    for sid in sids:
        p = preds[preds.seriesuid == sid]
        c, prob = p[XYZ].to_numpy(float), p.probability.to_numpy(float)
        a = ann[ann.seriesuid == sid]
        g = ign[ign.seriesuid == sid]
        alive = np.ones(len(c), bool)
        nod = []
        for xyz, d in zip(a[XYZ].to_numpy(float), a.diameter_mm):
            hit = ((c - xyz) ** 2).sum(1) < (d / 2) ** 2
            nod.append(prob[hit].max() if hit.any() else -np.inf)
            alive &= ~hit
        for xyz, d in zip(world_list(g.center_world_xyz), g.diameter_mm):
            alive &= ~(((c - xyz) ** 2).sum(1) < (d / 2) ** 2)
        rec[sid] = {"nod_p": np.array(nod), "nod_d": a.diameter_mm.to_numpy(float), "fp": prob[alive]}
    return rec


def froc(records):
    """返回 (fps, sens, thresholds)，与官方 computeFROC 等价（起点 (0, 0)）。"""
    nod = np.concatenate([r["nod_p"] for r in records])
    fp = np.concatenate([r["fp"] for r in records])
    thr = np.unique(np.concatenate([nod[np.isfinite(nod)], fp]))[::-1]
    sens = (len(nod) - np.searchsorted(np.sort(nod), thr)) / len(nod)
    fps = (len(fp) - np.searchsorted(np.sort(fp), thr)) / len(records)
    return np.r_[0, fps], np.r_[0, sens], np.r_[np.inf, thr]


def froc_metrics(records):
    records = list(records.values()) if isinstance(records, dict) else records
    fps, sens, thr = froc(records)
    pts = np.interp(FP_POINTS, fps, sens)
    m = {"cpm": float(pts.mean()), **{f"sens@{f:g}": float(s) for f, s in zip(FP_POINTS, pts)}}
    nod = np.concatenate([r["nod_p"] for r in records])
    d = np.concatenate([r["nod_d"] for r in records])
    for target in (1, 4):
        t = thr[np.flatnonzero(fps <= target)[-1]]
        for name, (lo, hi) in SIZE_BINS.items():
            g = (d >= lo) & (d < hi)
            m[f"sens{target}fp_{name}"] = float((nod[g] >= t).mean())
    return m


def bootstrap(records, patients, fn, n, seed=0):
    """按患者重抽样 n 次；records 为 {sid: rec}，patients 为 {sid: patient_id}。"""
    by_pid = {}
    for sid, pid in patients.items():
        by_pid.setdefault(pid, []).append(records[sid])
    pids = sorted(by_pid)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append(fn([r for k in rng.integers(len(pids), size=len(pids)) for r in by_pid[pids[k]]]))
    return pd.DataFrame(out)


def official(cfg, preds_csv, sids, out_dir):
    """调用官方评测脚本；读取其写出的 froc_<name>.txt（fps, sens, threshold）计算 CPM。"""
    script = Path(cfg.paths.get("luna16_evaluation_script", str(OFFICIAL))).resolve()
    if not script.is_file():
        raise FileNotFoundError(f"LUNA16 官方评测脚本不存在：{script}；请配置 paths.luna16_evaluation_script")
    out_dir.mkdir(parents=True, exist_ok=True)
    sid_csv = out_dir / "seriesuids.csv"
    sid_csv.write_text("\n".join(sids) + "\n")
    subprocess.run([sys.executable, str(script), cfg.paths.luna16_annotations, cfg.paths.luna16_excluded,
                    str(sid_csv), str(preds_csv), str(out_dir) + "/"], check=True, cwd=script.parent)
    curve = np.loadtxt(out_dir / f"froc_{Path(preds_csv).stem}.txt", delimiter=",")
    pts = np.interp(FP_POINTS, curve[:, 0], curve[:, 1])
    return {"cpm": float(pts.mean()), **{f"sens@{f:g}": float(s) for f, s in zip(FP_POINTS, pts)}}


def evaluate(run, split, n_boot):
    cfg = OmegaConf.load(run / "config.yaml")
    splits = pd.read_csv(Path(cfg.paths.meta) / "splits_det.csv")
    s = splits[splits.split == split]
    preds_csv = run / f"preds_{split}.csv"
    rec = scan_records(cfg, pd.read_csv(preds_csv), list(s.seriesuid))
    boot = bootstrap(rec, dict(zip(s.seriesuid, s.patient_id)), froc_metrics, n_boot)
    write_csv(run / f"bootstrap_{split}.csv", boot)
    metrics = {"self": froc_metrics(rec), "ci95": {k: [float(boot[k].quantile(0.025)), float(boot[k].quantile(0.975))] for k in boot},
               "official": official(cfg, preds_csv, list(s.seriesuid), run / f"official_{split}")}
    write_json(run / f"metrics_{split}.json", metrics)
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    m = evaluate(Path(args.run), args.split, args.bootstrap)
    print(f"CPM self {m['self']['cpm']:.4f} / official {m['official']['cpm']:.4f}")
    print({k: round(v, 4) for k, v in m["self"].items()})
