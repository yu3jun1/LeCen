"""随机数流与确定性设置。

init 1000（共享模块初始化）；recovery_init 1001（恢复 decoder，仅 D1）；
data 2000（采样器独立 Generator，每个样本的增强种子 = hash(2000, epoch, index)）；临床划分 2026。
"""
import logging
import warnings
from contextlib import contextmanager

import numpy as np
import torch


def set_determinism():
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    logging.captureWarnings(True)  # 无法确定性的算子警告写入运行日志
    warnings.simplefilter("default")


def generator(seed):
    return torch.Generator().manual_seed(seed)


def sample_seed(base, epoch, index):
    """与 worker 数量无关的逐样本种子。"""
    return int(np.random.SeedSequence([base, epoch, index]).generate_state(1)[0])


@contextmanager
def seeded(seed):
    """在独立随机流中构建模块：不影响、也不受全局随机状态影响。"""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield
