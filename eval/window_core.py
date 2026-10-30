"""滑窗中心区域过滤（infer.window_core）的验证集对照。

同一权重分别关闭、开启 window_core 推理验证集，比较自实现 FROC 与按尺寸的信息丢失链（p > infer.p_min，同训练期验证）。
只用验证集；结果写入 <run>/mech/window_core_val.json，不改动 p_min.json 与 preds_*.csv。

python -m eval.window_core --run runs/D1/<run_id>
"""
import argparse
from pathlib import Path

import pandas as pd

from common.io import write_json
from engine.infer_det import load_detector, predict, preds_frame
from eval.froc import froc_metrics, scan_records
from eval.recall_chain import luna_chain, summarize


def main(run):
    cfg, model = load_detector(run, "cuda")
    s = pd.read_csv(Path(cfg.paths.meta) / "splits_det.csv")
    ids = s[s.split == "val"].seriesuid
    out = {}
    for core in (False, True):
        cfg.infer.window_core = core
        stages = predict(model, cfg, ids, Path(cfg.paths.processed) / "luna16", "cuda")
        rec = scan_records(cfg, preds_frame(stages, cfg.infer.p_min), list(stages))
        out[f"window_core={core}"] = {"froc": froc_metrics(rec),
                                      "recall_chain": summarize(luna_chain(cfg, stages, rec)).to_dict("records")}
    write_json(run / "mech" / "window_core_val.json", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    res = main(Path(ap.parse_args().run))
    for k, v in res.items():
        f = v["froc"]
        print(k, {m: round(f[m], 4) for m in ("cpm", "sens@0.125", "sens@1", "sens@8", "sens1fp_3to6")})
