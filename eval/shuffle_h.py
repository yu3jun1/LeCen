"""打乱 h 测试。

固定的验证 patch 集合（以每个 has_mask 实例为中心，无增强）上，对匹配且 has_mask 的病灶，
分别把正确的 h 与错位排列（无不动点）的 h 送入恢复 decoder；报告 mask Dice 与强度 L1（乘 1400 换算为 HU）。
模板基线：不看 h，对所有病灶输出同一个预测（这些病灶目标的逐体素平均 mask 与平均强度，在评估集上取得，偏向模板）。
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
    """所有病灶目标的逐体素平均：mask logit = 平均 mask − 0.5（平均 ≥ 0.5 处为前景），强度 = 平均强度。"""
    x = torch.stack([t[0] for t in targets]).mean(0)
    msk = torch.stack([t[1] for t in targets]).mean(0)
    return torch.stack([msk - 0.5, x])


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
    h = torch.stack(hs).float()
    keep = [k for k, t in enumerate(targets) if t[2].sum() > 0]
    res = {}
    template = template_prediction([targets[k] for k in keep])
    for name, pred in (("true", recovery(h).float()), ("shuffled", recovery(h.roll(1, 0)).float()),
                       ("template", template.expand(len(targets), *template.shape))):
        m = np.array([lesion_metrics(pred[k], *targets[k], cfg.model.recovery.dilate_vox) for k in keep])
        res[name] = {"dice": float(m[:, 0].mean()), "intensity_l1_hu": float(m[:, 1].mean())}
    write_json(run / "mech" / "shuffle_h.json", {"n_lesions": len(keep), **res})
    print(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    main(Path(ap.parse_args().run))
