"""检测损失（各层都计算，求和）。

存在性：sigmoid focal（α=0.25, γ=2）。匹配的 query 目标 1，未匹配目标 0；未匹配且中心落在忽略区域内的 query 权重为 0；
已匹配正例优先。按 max(GT 数, 1) 归一化。
框：匹配 query 的 L1（归一化中心与 log 尺寸）+ 3D GIoU 损失；无 GT 时为 0。
临床部分标注（partial）：已标注肿瘤正常匹配；未匹配但与某个已标注肿瘤 3D IoU > 0.3 的 query 作为负例；其余未匹配 query 权重为 0。
"""
import torch
import torch.nn.functional as F

from common.boxes import giou3d, iou3d, norm_box


def sigmoid_focal(logit, target, alpha, gamma):
    p = logit.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    return (alpha * target + (1 - alpha) * (1 - target)) * (1 - p_t) ** gamma * ce


def in_ignore(centers, ignore):
    return ((((centers[:, None] - ignore[None, :, :3]) / ignore[None, :, 3:]) ** 2).sum(-1) <= 1).any(1)


def layer_loss(layer, batch, matches, cfg, patch_zyx, partial):
    focal = cfg.loss.exist_focal
    l_exist, l_l1, l_giou, n_gt = 0.0, 0.0, 0.0, 0
    for b, (row, col) in enumerate(matches):
        logit, box, gt = layer["exist_logit"][b], layer["box"][b], batch["boxes"][b]
        target = torch.zeros_like(logit)
        target[row] = 1
        if partial:
            weight = (iou3d(box[:, None].detach(), gt[None]) > 0.3).any(1).float()
        else:
            weight = (~in_ignore(box[:, :3].detach(), batch["ignore"][b])).float()
        weight[row] = 1
        l_exist = l_exist + (sigmoid_focal(logit, target, focal.alpha, focal.gamma) * weight).sum()
        if len(row):
            l_l1 = l_l1 + F.l1_loss(norm_box(box[row], patch_zyx), norm_box(gt[col], patch_zyx), reduction="sum")
            l_giou = l_giou + (1 - giou3d(box[row], gt[col])).sum()
        n_gt += len(gt)
    n = max(n_gt, 1)
    return l_exist / n, l_l1 / n, l_giou / n


def det_losses(out, batch, layer_matches, cfg, patch_zyx, partial=False):
    """返回加权后的 Σ_layers(w_exist·L_exist + w_l1·L_l1 + w_giou·L_giou) 与各项（各层求和）。"""
    w = cfg.loss.w
    terms = [layer_loss(l, batch, m, cfg, patch_zyx, partial) for l, m in zip(out["layers"], layer_matches)]
    exist, l1, giou = (sum(t[i] for t in terms) for i in range(3))
    return w.exist * exist + w.l1 * l1 + w.giou * giou, {"exist": exist, "l1": l1, "giou": giou}
