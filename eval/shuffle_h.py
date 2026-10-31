"""打乱 h 测试。

固定的验证 patch 集合（以每个 has_mask 实例为中心，无增强）上，对匹配且 has_mask 的病灶，
分别把正确的 h 与错位排列（无不动点）的 h 送入恢复 decoder；报告 mask Dice 与强度 L1（乘 1400 换算为 HU）。
模板基线：不看 h，对所有病灶输出同一个预测（按逐体素 valid_mask 加权的平均 mask 与强度，在验证集上取得，偏向模板）。
true 需要同时明显优于 shuffled 与 template，才说明 h 编码了病灶特异的信息。

python -m eval.shuffle_h --run runs/D1/<run_id>
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from common.io import write_json
from data.det_dataset import collate, luna_dataset, to_device
from engine.infer_det import load_detector
from losses.matching import match_layers
from losses.recovery_loss import dilate3d, recovery_inputs
from models.recovery import RecoveryDecoder


def template_prediction(targets):
    """逐体素按 valid_mask 加权；平均 mask > 0.5 为前景，无观测处为背景/零强度。"""
    if not targets:
        raise ValueError("Template baseline requires at least one valid recovery target")
    valid = torch.stack([t[2] for t in targets]).float()
    count = valid.sum(0)
    if not (count > 0).any():
        raise ValueError("Template baseline has no valid recovery voxels")
    denominator = torch.where(count > 0, count, torch.ones_like(count))

    def mean(channel):
        values = torch.stack([t[channel] for t in targets]).float()
        weighted = torch.where(valid > 0, values * valid, torch.zeros_like(values))
        return weighted.sum(0) / denominator

    return torch.stack([mean(1) - 0.5, mean(0)])


def comparison_inputs(hs, targets):
    """Only valid target/h pairs can participate in a nontrivial cyclic shuffle."""
    if len(hs) != len(targets):
        raise ValueError("Recovery features and targets must have equal lengths")
    keep = [k for k, target in enumerate(targets) if target[2].sum() > 0]
    if len(keep) < 2:
        raise ValueError(f"shuffle-h requires at least two valid recovery targets; found {len(keep)}")
    return torch.stack([hs[k] for k in keep]).float(), [targets[k] for k in keep]


def lesion_metrics(pred, x, msk, val, dilate):
    p = (pred[0] > 0).float() * val
    m = msk * val
    region = dilate3d(msk, dilate) * val
    dice = 2 * (p * m).sum() / (p.sum() + m.sum())
    l1 = ((pred[1] - x).abs() * region).sum() / region.sum() * 1400
    return float(dice), float(l1)


@torch.no_grad()
def main(run):
    dev = "cuda"
    cfg, model = load_detector(run, dev)
    recovery = RecoveryDecoder(cfg.model.dim).to(dev).eval()
    recovery.load_state_dict(torch.load(run / "checkpoints" / "best.pt", map_location="cpu")["recovery"])
    patch = torch.tensor(list(cfg.patch_zyx), dtype=torch.float32, device=dev)
    ds = luna_dataset(cfg, "val", augment=False)
    samples = [ds.patch_at(c, center, None) for c in map(ds.case, range(len(ds.ids))) for center in c.centers[c.has_mask]]
    hs, targets = [], []
    for i in range(0, len(samples), cfg.train.batch):
        batch = to_device(collate(samples[i:i + cfg.train.batch]), dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.train.amp):
            out = model(batch["image"], batch["tokens"], batch["token_xyz"], batch)
        h, t = recovery_inputs(out, batch, match_layers(out, batch["boxes"], cfg, patch)[-1], cfg, patch)
        hs += h
        targets += t
    h, targets = comparison_inputs(hs, targets)
    res = {}
    template = template_prediction(targets)
    for name, pred in (("true", recovery(h).float()), ("shuffled", recovery(h.roll(1, 0)).float()),
                       ("template", template.expand(len(targets), *template.shape))):
        m = np.array([lesion_metrics(p, *target, cfg.model.recovery.dilate_vox) for p, target in zip(pred, targets)])
        res[name] = {"dice": float(m[:, 0].mean()), "intensity_l1_hu": float(m[:, 1].mean())}
    write_json(run / "mech" / "shuffle_h.json", {"n_lesions": len(targets), **res})
    print(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    main(Path(ap.parse_args().run))
