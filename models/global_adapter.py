"""G 的 adapter 与位置编码。G = Linear(LayerNorm(tokens)) + PE(token_xyz)；查询位置使用同一个 PE。"""
import math

import torch
import torch.nn as nn


class PosEnc(nn.Module):
    """世界坐标 (mm) 除以 500 后做每轴 32 个频率的 sin/cos（共 192 维），再 Linear 到 D。"""

    def __init__(self, dim, n_freq=32, scale_mm=500.0):
        super().__init__()
        self.register_buffer("freq", math.pi * 2.0 ** torch.linspace(0, 8, n_freq), persistent=False)
        self.scale = scale_mm
        self.proj = nn.Linear(6 * n_freq, dim)

    def forward(self, xyz):
        a = (xyz.float() / self.scale)[..., None] * self.freq
        return self.proj(torch.cat([a.sin(), a.cos()], -1).flatten(-2))


class GlobalAdapter(nn.Module):
    def __init__(self, dim, pe, token_dim=512):
        super().__init__()
        self.norm = nn.LayerNorm(token_dim)
        self.proj = nn.Linear(token_dim, dim)
        self.pe = pe

    def forward(self, tokens, token_xyz):
        return self.proj(self.norm(tokens.float())) + self.pe(token_xyz)
