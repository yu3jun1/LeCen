"""生存评测。

风险分数：所有模型统一使用 risk = 1 − Ŝ(t*)。
Harrell C-index：concordance_index_censored；IPCW C-index：concordance_index_ipcw(survival_train, survival_test, risk, tau)，
删失分布用训练集估计；IBS：与 sksurv integrated_brier_score(survival_train, survival_test, S_pred(times), times) 相同的 IPCW Brier 积分，
删失分布用训练集估计；sksurv 要求 times 落在测试随访范围内，而 times 为预先固定区间内的 50 个等距点、
bootstrap 重抽样的随访范围随样本变化，因此直接按公式计算；校准：在 t* 处按预测风险分为 3 组，比较各组平均预测生存概率与 KM 观测值（描述性）。
预先固定：t* = 730 天，若训练集中位随访（反向 KM）不足则改为该中位随访时间；τ = min(t*, 训练集随访时间的 80 分位数)；
IBS 区间 [训练集最早事件时间, τ]。置信区间按患者 bootstrap 2,000 次。

外部测试队列（split = external）使用开发队列训练集的 t*、τ、IBS 区间与删失分布。

python -m eval.surv_eval --run runs/S5/<run_id> --final   # 测试集与外部测试队列各评一次
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from omegaconf import OmegaConf
from scipy.integrate import trapezoid
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv

from common.io import read_npz, register_final, write_json


def fixed_times(time, event, s):
    median_fu = KaplanMeierFitter().fit(time, 1 - event).median_survival_time_
    t_star = s.t_star_days if median_fu >= s.t_star_days else float(median_fu)
    tau = min(t_star, float(np.quantile(time, s.ipcw_quantile)))
    return t_star, tau, np.linspace(time[event == 1].min(), tau, s.ibs_points)


def calibration(time, event, risk, t_star):
    groups = pd.qcut(risk, 3, labels=["low", "mid", "high"])
    rows = []
    for g in ["low", "mid", "high"]:
        m = np.asarray(groups == g)
        km = KaplanMeierFitter().fit(time[m], event[m])
        rows.append({"group": g, "n": int(m.sum()), "pred_surv": float(1 - risk[m].mean()),
                     "km_surv": float(km.survival_function_at_times(t_star).iloc[0])})
    return rows


def integrated_brier(y_train, time, event, surv, times):
    cens = CensoringDistributionEstimator().fit(y_train)
    case = (time[:, None] <= times[None]) & event.astype(bool)[:, None]
    control = time[:, None] > times[None]
    g_y = np.ones_like(time)
    used = case.any(1)
    g_y[used] = cens.predict_proba(time[used])
    bs = (surv ** 2 * case / g_y[:, None] + (1 - surv) ** 2 * control / cens.predict_proba(times)[None]).mean(0)
    return trapezoid(bs, times) / (times[-1] - times[0])


def metrics(y_train, time, event, risk, surv, times, tau):
    y = Surv.from_arrays(event.astype(bool), time)
    return {"cindex_harrell": concordance_index_censored(event.astype(bool), time, risk)[0],
            "cindex_ipcw": concordance_index_ipcw(y_train, y, risk, tau)[0],
            "ibs": integrated_brier(y_train, time, event, surv, times)}


def load(run, split):
    cfg = OmegaConf.load(run / "config.yaml")
    meta = Path(cfg.paths.clinical_meta)
    clin = pd.read_csv(meta / "clinical.csv", dtype={"patient_id": str}).merge(
        pd.read_csv(meta / "splits_clin.csv", dtype={"patient_id": str})[["patient_id", "split"]], on="patient_id")
    tr = clin[clin.split == "train"]
    p = read_npz(run / f"preds_{split}.npz")
    df = clin.set_index("patient_id").loc[p["patient_id"]]
    return cfg, Surv.from_arrays(tr.event.astype(bool), tr.time_days), df.time_days.to_numpy(float), df.event.to_numpy(int), p


def evaluate(run, split, n_boot, seed=0):
    cfg, y_train, time, event, p = load(run, split)
    risk, surv, times, tau, t_star = p["risk"], p["surv"], p["times"], float(p["tau"]), float(p["t_star"])
    point = metrics(y_train, time, event, risk, surv, times, tau)
    rng = np.random.default_rng(seed)
    boot = pd.DataFrame([metrics(y_train, time[i], event[i], risk[i], surv[i], times, tau)
                         for i in (rng.integers(len(time), size=len(time)) for _ in range(n_boot))])
    out = {"point": {k: float(v) for k, v in point.items()},
           "ci95": {k: [float(boot[k].quantile(0.025)), float(boot[k].quantile(0.975))] for k in boot},
           "calibration": calibration(time, event, risk, t_star), "t_star": t_star, "tau": tau}
    write_json(run / f"metrics_{split}.json", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    run = Path(args.run)
    if args.final:
        register_final(OmegaConf.load(run / "config.yaml").name, run, run / "best.pt")
    for split in ("test", "external") if args.final else ("val",):
        print(split, evaluate(run, split, args.bootstrap)["point"])
