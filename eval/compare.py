"""配对差值与判定。

测试集按患者 bootstrap 2,000 次；每次在同一组重抽样患者上同时计算两个配置的指标，得到配对差值分布（a − b）。
“越大越好”的指标直接比较，IBS 取负值后同样处理。差值 95% 区间：全部 > 0 → 初步支持增益；跨过 0 → 证据不足以确认增益；
全部 < 0 → 提示性能下降。bootstrap 只反映测试患者抽样的不确定性，不包含重新训练与换种子带来的波动。

python -m eval.compare --a runs/D1/<id> --b runs/D2/<id> --metrics cpm,sens1fp_3to6 --bootstrap 2000
python -m eval.compare --a runs/S5/<id> --b runs/S2/<id> --metrics cindex_ipcw --bootstrap 2000
python -m eval.compare --a runs/S5/<id> --b runs/S1/<id> --metrics cindex_ipcw --split external
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from common.io import ROOT, write_json
from eval import surv_eval
from eval.froc import froc_metrics, scan_records

NEGATE = {"ibs"}


def verdict(lo, hi):
    if lo > 0:
        return "初步支持增益"
    if hi < 0:
        return "提示性能下降"
    return "证据不足以确认增益"


def det_samplers(run_a, run_b):
    cfg = OmegaConf.load(run_a / "config.yaml")
    s = pd.read_csv(Path(cfg.paths.meta) / "splits_det.csv")
    s = s[s.split == "test"]
    rec = [scan_records(cfg, pd.read_csv(r / "preds_test.csv"), list(s.seriesuid)) for r in (run_a, run_b)]
    by_pid = s.groupby("patient_id").seriesuid.apply(list)

    def fn(r, pids):
        return froc_metrics([r[sid] for p in pids for sid in by_pid[p]])
    return list(by_pid.index), lambda pids: (fn(rec[0], pids), fn(rec[1], pids))


def surv_samplers(run_a, run_b, split):
    loaded = [surv_eval.load(r, split) for r in (run_a, run_b)]
    pids = list(loaded[0][4]["patient_id"])

    def fn(l, idx):
        _, y_tr, time, event, p = l
        return surv_eval.metrics(y_tr, time[idx], event[idx], p["risk"][idx], p["surv"][idx], p["times"], float(p["tau"]))
    order = [{pid: i for i, pid in enumerate(l[4]["patient_id"])} for l in loaded]
    return pids, lambda ps: tuple(fn(l, np.array([o[p] for p in ps])) for l, o in zip(loaded, order))


def compare(run_a, run_b, metric_names, n_boot, split="test", seed=0):
    """split：检测只有 test；生存为 test 或 external（外部测试队列）。"""
    if (run_a / "preds_test.csv").exists():
        pids, both = det_samplers(run_a, run_b)
    else:
        pids, both = surv_samplers(run_a, run_b, split)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        ma, mb = both([pids[k] for k in rng.integers(len(pids), size=len(pids))])
        diffs.append({k: (-1 if k in NEGATE else 1) * (ma[k] - mb[k]) for k in metric_names})
    d = pd.DataFrame(diffs)
    out = {}
    for k in metric_names:
        lo, hi = float(d[k].quantile(0.025)), float(d[k].quantile(0.975))
        out[k] = {"diff_mean": float(d[k].mean()), "ci95": [lo, hi], "verdict": verdict(lo, hi)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--split", choices=["test", "external"], default="test")
    args = ap.parse_args()
    a, b = Path(args.a), Path(args.b)
    res = compare(a, b, args.metrics.split(","), args.bootstrap, args.split)
    write_json(ROOT / "runs" / "compare" / f"{a.parent.name}-{a.name}_vs_{b.parent.name}-{b.name}_{args.split}.json",
               {"a": str(a), "b": str(b), "split": args.split, "n_boot": args.bootstrap, "results": res})
    for k, v in res.items():
        print(f"{k}: Δ={v['diff_mean']:+.4f}  95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  {v['verdict']}")
