"""逐层存在性与框细化 head：预测 Δ中心（相对参考框尺寸）与 Δlog 尺寸。"""
import math

import torch
import torch.nn as nn


class DetHead(nn.Module):
    def __init__(self, dim, prior=0.01):
        super().__init__()
        self.exist = nn.Linear(dim, 1)
        self.box = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, 6))
        nn.init.constant_(self.exist.bias, -math.log((1 - prior) / prior))
        nn.init.zeros_(self.box[-1].weight)
        nn.init.zeros_(self.box[-1].bias)

    def forward(self, q, center, size):
        d = self.box(q).float()
        box = torch.cat([center + d[..., :3] * size, size * d[..., 3:].exp()], -1)
        return self.exist(q).squeeze(-1).float(), box
