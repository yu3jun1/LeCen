"""同配置跑 200 步两次：数据序列一致，损失曲线差异在容差内。完整训练部分需要 GPU 与预处理数据。"""
import itertools

import numpy as np
import pandas as pd

from common.io import ROOT, load_config
from data.det_dataset import PatchSampler
from engine.train_det import train

LOSS_RTOL = 1e-2


def test_sampler_sequence():
    seq = [list(itertools.islice(iter(PatchSampler(50, [1, 3, 7, 20], 0.5, 2000)), 500)) for _ in range(2)]
    assert seq[0] == seq[1]


def test_train_200_steps():
    cfg = load_config(ROOT / "configs" / "det" / "D1.yaml", ["train.iters=200", "train.val_every=1000000"])
    logs = [pd.read_csv(train(cfg) / "train_log.csv") for _ in range(2)]
    cols = ["hm", "size", "exist", "l1", "giou", "local"]
    assert np.allclose(logs[0][cols].to_numpy(), logs[1][cols].to_numpy(), rtol=LOSS_RTOL, atol=1e-4)
