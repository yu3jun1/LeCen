"""匈牙利匹配。每个样本只对最终层输出匹配一次；aux_match: final 时各层辅助损失与恢复损失复用该结果，
per_layer 时辅助检测损失逐层匹配，恢复损失仍用最终层。"""
import torch
from scipy.optimize import linear_sum_assignment

from common.boxes import giou3d, norm_box


def focal_cost(p, alpha, gamma):
    pos = alpha * (1 - p) ** gamma * -(p + 1e-8).log()
    neg = (1 - alpha) * p ** gamma * -(1 - p + 1e-8).log()
    return pos - neg


def hungarian_match(exist_logit, box, gt_boxes, patch_zyx, w, focal):
    """exist_logit (N,)，box (N, 6) 中心 + 尺寸（patch 体素），gt_boxes (M, 6)。返回 (query 下标, GT 下标)。"""
    if gt_boxes.shape[0] == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    p = exist_logit.float().sigmoid()
    c_cls = focal_cost(p, focal.alpha, focal.gamma)
    c_l1 = torch.cdist(norm_box(box.float(), patch_zyx), norm_box(gt_boxes, patch_zyx), p=1)
    c_giou = -giou3d(box.float()[:, None], gt_boxes[None])
    cost = w.cls * c_cls[:, None] + w.l1 * c_l1 + w.giou * c_giou
    row, col = linear_sum_assignment(cost.detach().float().cpu())
    return torch.as_tensor(row, dtype=torch.long), torch.as_tensor(col, dtype=torch.long)


def match_layers(out, gt_boxes, cfg, patch_zyx):
    """返回每层的逐样本匹配列表；最后一项为最终层匹配。"""
    def match(layer):
        return [hungarian_match(layer["exist_logit"][b], layer["box"][b], g, patch_zyx, cfg.loss.match_cost, cfg.loss.exist_focal)
                for b, g in enumerate(gt_boxes)]

    final = match(out["layers"][-1])
    if cfg.loss.aux_match == "final":
        return [final] * len(out["layers"])
    return [match(l) for l in out["layers"][:-1]] + [final]
