"""离散时间生存损失：用训练集事件时间分位数确定 N_t 个区间边界（pycox LogisticHazard.label_transform），
NLLLogistiHazardLoss 输入原始 logits；S(t_k) = cumprod(1 − sigmoid(logits))，再插值到任意时间。"""
import numpy as np
import torch
from pycox.models import LogisticHazard
from pycox.models.loss import NLLLogistiHazardLoss

nll_logistic_hazard = NLLLogistiHazardLoss()


def fit_label_transform(n_t, time, event):
    lt = LogisticHazard.label_transform(n_t, scheme="quantiles")
    lt.fit(time.astype(np.float32), event.astype(np.float32))
    return lt


def survival(logits):
    return torch.cumprod(1 - torch.sigmoid(logits), -1)


def interp_survival(surv, cuts, times):
    """surv (n, N_t) 在 cuts 上的取值 → (n, len(times))。"""
    return np.stack([np.interp(times, cuts, s) for s in surv])
