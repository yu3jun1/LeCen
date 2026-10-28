"""中心热图与候选提取。输入 P[L]，输出 hm_logit (B, 1, …) 与 size (B, 3, …)（log 尺寸，单位 patch 体素）。"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.boxes import size_from_log
from common.sampling import from_level


class HeatmapHead(nn.Module):
    def __init__(self, dim, prior=0.1):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv3d(dim, dim, 3, 1, 1), nn.GroupNorm(16, dim), nn.ReLU(inplace=True))
        self.hm = nn.Conv3d(dim, 1, 1)
        self.size = nn.Conv3d(dim, 3, 1)
        nn.init.constant_(self.hm.bias, -math.log((1 - prior) / prior))

    def forward(self, p):
        y = self.conv(p)
        return self.hm(y), self.size(y)

    @staticmethod
    def candidates(hm, size, stride, n, kernel):
        """sigmoid → k³ 最大池化去重 → top-N；层坐标换回 patch 体素坐标 u = (u_L + 0.5)·s_L − 0.5。
        候选点大多不在正例中心，其 log 尺寸未受监督，经 size_from_log 限制范围。"""
        h = hm.detach().float().sigmoid()
        score = (h * (h == F.max_pool3d(h, kernel, 1, kernel // 2))).flatten(1)
        idx = score.topk(n, 1).indices
        level_zyx = torch.stack(torch.unravel_index(idx, hm.shape[2:]), -1).float()
        log_size = size.detach().float().flatten(2).gather(2, idx[:, None].expand(-1, 3, -1)).transpose(1, 2)
        patch = torch.tensor(hm.shape[2:], dtype=torch.float32, device=hm.device) * stride
        return from_level(level_zyx, stride), size_from_log(log_size, patch)
