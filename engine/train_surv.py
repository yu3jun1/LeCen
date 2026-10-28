"""生存模型训练（S0–S5）。

python -m engine.train_surv --cfg configs/clin/S5.yaml slots.extractor=runs/D1/<run_id>

S0：lifelines CoxPH（penalizer 0.1），输入临床变量 + 治疗 one-hot。
其余：AdamW（lr 1e-3，wd 1e-4），batch 32，最多 200 epoch，以验证集损失早停（patience 20）。
临床变量标准化、缺失处理、时间分箱只在训练集上拟合。验证集与测试集输出 risk = 1 − Ŝ(t*) 与 IBS 时间点上的 Ŝ。
"""
import argparse
import copy

import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter

from common.io import data_hashes, load_config, make_run, read_json, write_npz
from common.seed import generator
from data.clin_dataset import cohort, load_table, take
from eval.surv_eval import fixed_times
from losses.surv_loss import fit_label_transform, interp_survival, nll_logistic_hazard, survival
from models.clinical import ClinicalModel


def fit_cox(cfg, df, x, times):
    """治疗 one-hot 以首个类别为参照；只保留训练集上方差非零的列（例如治疗只有一类时不进入模型）。"""
    tr = (df.split == "train").to_numpy()
    cols = pd.concat([x, pd.get_dummies(df.treatment, prefix="treatment", drop_first=True, dtype=float)], axis=1)
    cols = cols.loc[:, cols[tr].std() > 0]
    cph = CoxPHFitter(penalizer=cfg.cox.penalizer)
    cph.fit(pd.concat([cols[tr], df.loc[tr, ["time_days", "event"]]], axis=1), "time_days", "event")
    return lambda m: cph.predict_survival_function(cols[m], times=times).T.to_numpy()


def fit_neural(cfg, df, x, times, run):
    torch.manual_seed(cfg.seed)
    data, n_treat = cohort(cfg, df, x)
    tr, va = np.flatnonzero(df.split == "train"), np.flatnonzero(df.split == "val")
    lt = fit_label_transform(cfg.surv.n_t, data["time"][tr], data["event"][tr])
    idx, ev = (torch.tensor(a) for a in lt.transform(data["time"], data["event"]))
    model = ClinicalModel(cfg.model, data["pool_g"].shape[1], data["c"].shape[1], n_treat, len(lt.cuts))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
    g = generator(cfg.seed)

    def loss_on(i):
        return nll_logistic_hazard(model(take(data, i)), idx[i], ev[i])

    best, best_state, wait, rows = np.inf, None, 0, []
    for epoch in range(cfg.train.max_epochs):
        model.train()
        for b in torch.randperm(len(tr), generator=g).split(cfg.train.batch):
            opt.zero_grad(set_to_none=True)
            loss = loss_on(tr[b.numpy()])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(loss_on(va))
        rows.append({"epoch": epoch, "train_loss": float(loss), "val_loss": val})
        if val < best:
            best, best_state, wait = val, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg.train.patience:
                break
    pd.DataFrame(rows).to_csv(run / "train_log.csv", index=False)
    model.load_state_dict(best_state)
    torch.save(best_state, run / "best.pt")
    model.eval()

    @torch.no_grad()
    def predict(m):
        return interp_survival(survival(model(take(data, np.flatnonzero(m)))).numpy(), lt.cuts, times)
    return predict


def main(cfg):
    df, x = load_table(cfg)
    tr = df[df.split == "train"]
    t_star, tau, ibs_times = fixed_times(tr.time_days.to_numpy(float), tr.event.to_numpy(int), cfg.surv)
    times = np.r_[t_star, ibs_times]
    hashes = data_hashes(cfg)
    if cfg.model.type != "cox":
        hashes.update(read_json(f"{cfg.slots.extractor}/slots.json"))
    run = make_run(cfg, hashes)
    predict = fit_cox(cfg, df, x, times) if cfg.model.type == "cox" else fit_neural(cfg, df, x, times, run)
    for split in ("val", "test"):
        m = (df.split == split).to_numpy()
        s = predict(m)
        write_npz(run / f"preds_{split}.npz", patient_id=df.patient_id[m].to_numpy(str), risk=1 - s[:, 0], surv=s[:, 1:],
                  times=ibs_times, t_star=np.array(t_star), tau=np.array(tau))
    return run


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    args, rest = ap.parse_known_args()
    print(main(load_config(args.cfg, rest)))
