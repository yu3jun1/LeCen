"""恢复损失（仅 D1）。

目标在 GT 框 × crop_scale 的范围内重采样为 out³：归一化强度（三线性，增强后输入）、实例 mask（最近邻）、
有效体素 = valid_mask（最近邻）× 采样点落在 patch 内。
每个病灶：mask 的 BCE（按有效体素归一化）+ Dice；强度 L1 只在膨胀后的 mask 区域内计算。
总恢复损失 = 所有有效病灶的平均；有效体素为 0 的病灶不参与平均并计数；无有效病灶时为 0。
"""
import torch
import torch.nn.functional as F

from common.sampling import box_grid, inside_patch, sample


def recovery_targets(image, labels, valid_mask, center, size, label, patch_zyx, out, scale):
    """image (1, D, H, W)，labels / valid_mask (D, H, W)。返回 out³ 的 (x, msk, val)。"""
    pts = box_grid(center, size * scale, out)
    g = pts[None]
    x = sample(image[None], g, patch_zyx, "bilinear")
    msk = sample((labels == label).float()[None, None], g, patch_zyx, "nearest")
    val = sample(valid_mask.float()[None, None], g, patch_zyx, "nearest") * inside_patch(pts, patch_zyx)
    return tuple(t.reshape(out, out, out) for t in (x, msk, val))


def dilate3d(m, r):
    return F.max_pool3d(m[None, None], 2 * r + 1, 1, r)[0, 0]


def dice_loss(logit, m, v):
    p = logit.sigmoid()
    return 1 - (2 * (p * m * v).sum() + 1) / ((p * v).sum() + (m * v).sum() + 1)


def recovery_loss_one(pred, x, msk, val, dilate):
    n_valid = val.sum()
    if n_valid == 0:
        return None
    region = dilate3d(msk, dilate) * val
    l_mask = (F.binary_cross_entropy_with_logits(pred[0], msk, reduction="none") * val).sum() / n_valid + dice_loss(pred[0], msk, val)
    l_int = ((pred[1] - x).abs() * region).sum() / region.sum().clamp(min=1)
    return l_mask + l_int


def recovery_inputs(out, batch, matches, cfg, patch_zyx):
    """匹配且 has_mask 的病灶：最终层 h (M, D) 与各自的恢复目标。"""
    r = cfg.model.recovery
    hs, targets = [], []
    for b, (row, col) in enumerate(matches):
        keep = batch["has_mask"][b][col].cpu()
        for q, g in zip(row[keep], col[keep]):
            box = batch["boxes"][b][g]
            hs.append(out["layers"][-1]["h"][b, q])
            targets.append(recovery_targets(batch["image"][b], batch["labels"][b], batch["valid_mask"][b], box[:3], box[3:],
                                            batch["label_value"][b][g], patch_zyx, r.out_size, r.crop_scale))
    return hs, targets


def recovery_term(recovery, out, batch, matches, cfg, patch_zyx):
    """返回 (恢复损失, 参与平均的病灶数, 有效体素为 0 的病灶数)；本 batch 没有可恢复病灶时不调用 decoder。"""
    hs, targets = recovery_inputs(out, batch, matches, cfg, patch_zyx)
    if not hs:
        return 0.0, 0, 0
    pred = recovery(torch.stack(hs)).float()
    losses = [recovery_loss_one(p, *t, cfg.model.recovery.dilate_vox) for p, t in zip(pred, targets)]
    used = [l for l in losses if l is not None]
    loss = torch.stack(used).mean() if used else 0.0
    return loss, len(used), len(losses) - len(used)
