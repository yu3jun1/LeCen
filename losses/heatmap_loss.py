"""热图损失。

目标在热图所在层（stride s_L）网格上生成：正例中心换算为层坐标 u_L = (u + 0.5)/s_L − 0.5，取最近网格点为峰值；
σ（每轴，层体素）= max(尺寸_patch体素 / (6·s_L), 1)；多个高斯取最大值。
CenterNet 式 focal loss（α=2, β=4），按 max(正例数, 1) 归一化；忽略区域内的网格点不计负例损失。
尺寸：在正例中心所在网格点上对 log 尺寸做 L1；无正例时为 0。
"""
import torch
import torch.nn.functional as F

from common.sampling import from_level, to_level


def heatmap_targets(boxes, ignore, shape, stride):
    dev = boxes.device
    shape_t = torch.tensor(shape, device=dev)
    grid = torch.stack(torch.meshgrid(*[torch.arange(n, device=dev) for n in shape], indexing="ij"), -1).float()
    peak = to_level(boxes[:, :3], stride).round().long().clamp(min=torch.zeros_like(shape_t), max=shape_t - 1)
    sigma = (boxes[:, 3:] / (6 * stride)).clamp(min=1)
    tgt = torch.zeros(shape, device=dev)
    for k, s in zip(peak, sigma):
        tgt = torch.maximum(tgt, torch.exp(-(((grid - k) / s) ** 2).sum(-1) / 2))
    u = from_level(grid, stride)[..., None, :]
    ign = ((((u - ignore[:, :3]) / ignore[:, 3:]) ** 2).sum(-1) <= 1).any(-1)
    return tgt, ign, peak


def heatmap_loss(hm_logit, size_pred, batch, stride, alpha=2.0, beta=4.0):
    l_hm, l_size, n_pos = 0.0, 0.0, 0
    for b, boxes in enumerate(batch["boxes"]):
        x = hm_logit[b, 0].float()
        tgt, ign, peak = heatmap_targets(boxes, batch["ignore"][b], x.shape, stride)
        pos = tgt == 1
        p = x.sigmoid()
        l_pos = (1 - p) ** alpha * F.logsigmoid(x) * pos
        l_neg = p ** alpha * (1 - tgt) ** beta * F.logsigmoid(-x) * (~pos & ~ign)
        l_hm = l_hm - (l_pos.sum() + l_neg.sum())
        n_pos += len(boxes)
        if len(boxes):
            s = size_pred[b].float()[:, peak[:, 0], peak[:, 1], peak[:, 2]].T
            l_size = l_size + (s - boxes[:, 3:].log()).abs().sum()
    n = max(n_pos, 1)
    return l_hm / n, l_size / n
