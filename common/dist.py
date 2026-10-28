"""DDP。torchrun 启动时初始化 NCCL 进程组；直接 python -m 启动时为单卡。

train.batch 为全局 batch：采样器在每个 rank 上生成同一条全局样本序列，rank r 取全局下标 ≡ r (mod world) 的样本，
每个 rank 的 batch = train.batch / world，因此任意卡数下每个全局 batch 的样本都相同。
"""
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def setup():
    """返回本进程的 device。"""
    if "WORLD_SIZE" not in os.environ:
        return torch.device("cuda")
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    return torch.device("cuda", local)


def teardown():
    if dist.is_initialized():
        dist.destroy_process_group()


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world():
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main():
    return rank() == 0


def wrap(module, dev):
    return DistributedDataParallel(module, device_ids=[dev.index]) if dist.is_initialized() else module


def gather_dict(d):
    """各 rank 的 {key: value} 合并到每个 rank。"""
    if not dist.is_initialized():
        return d
    parts = [None] * world()
    dist.all_gather_object(parts, d)
    return {k: v for p in parts for k, v in p.items()}


def reduce_log(log, dev):
    """各 rank 的日志：计数（n_ 开头）求和，其余取平均。"""
    if not dist.is_initialized():
        return log
    t = torch.tensor(list(log.values()), dtype=torch.float64, device=dev)
    dist.all_reduce(t)
    return {k: v if k.startswith("n_") else v / world() for k, v in zip(log, t.tolist())}
