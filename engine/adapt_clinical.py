"""临床定位适配（仅在临床训练集与验证集上的肿瘤覆盖率不足时使用）。

python -m engine.adapt_clinical --extractor runs/D1/<run_id> adapt.iters=<步数>

每个 optimizer 步混合一个临床训练批次与一个 LUNA16 训练批次（1:1）；D1、D2 使用完全相同的 LUNA16 批次（同一数据种子）。
临床批次（部分标注）：已标注肿瘤匹配后计算存在性正例、L1 与 GIoU；与某个已标注肿瘤 3D IoU > 0.3 的未匹配 query 作为负例；
其余未匹配 query 存在性权重为 0。LUNA16 批次：完整检测损失；D1 额外计算恢复损失（只在 LUNA16 批次上）。
AMP 与梯度裁剪顺序同训练。适配后的权重存为新运行目录的 checkpoints/best.pt，提取 slot 时得到新的 extractor_hash。
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf

from common.hashing import file_sha1
from common.io import append_csv, make_run
from common.seed import set_determinism
from data.det_dataset import DetPatchDataset, loader, luna_dataset, to_device
from engine.train_det import build_optim, compute_loss, optimizer_step, save
from models.detector import Detector
from models.recovery import RecoveryDecoder


def main(src, overrides):
    set_determinism()
    dev = "cuda"
    cfg = OmegaConf.merge(OmegaConf.load(src / "config.yaml"), OmegaConf.from_dotlist(overrides))
    cfg.name, cfg.train.iters = f"{cfg.name}_adapt", cfg.adapt.iters
    ckpt = torch.load(src / "checkpoints" / "best.pt", map_location="cpu")
    run = make_run(cfg, {"source_run": str(src), "source_ckpt": file_sha1(src / "checkpoints" / "best.pt")})
    model = Detector(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(dev).train()
    recovery = None
    if cfg.model.recovery.enabled:
        recovery = RecoveryDecoder(cfg.model.dim)
        recovery.load_state_dict(ckpt["recovery"])
        recovery.to(dev).train()
    extra = list(recovery.parameters()) if recovery else []
    params = list(model.parameters()) + extra
    optimizer, scheduler, scaler = build_optim(cfg, model, extra)
    p = Path(cfg.paths.processed)
    splits = pd.read_csv(Path(cfg.paths.meta) / "splits_clin.csv", dtype={"patient_id": str})
    clin = DetPatchDataset(cfg, splits[splits.split == "train"].patient_id, p / "clinical", None, p / "clinical_targets", True)
    clin_batches = loader(cfg, clin, cfg.seeds.data)
    luna_batches = loader(cfg, luna_dataset(cfg, "train", augment=True), cfg.seeds.data)
    patch = torch.tensor(list(cfg.patch_zyx), dtype=torch.float32, device=dev)
    step, skipped = 0, 0
    while step < cfg.train.iters:
        bc, bl = to_device(next(clin_batches), dev), to_device(next(luna_batches), dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.train.amp):
            loss_c, log_c, _, _ = compute_loss(model, None, bc, cfg, patch, partial=True)
            loss_l, log_l, _, _ = compute_loss(model, recovery, bl, cfg, patch)
        scaler.scale(loss_c + loss_l).backward()
        stepped, grad_norm = optimizer_step(params, optimizer, scaler, scheduler, cfg.optim.clip_norm)
        if not stepped:
            skipped += 1
            continue
        step += 1
        append_csv(run / "train_log.csv", {"step": step, **{f"clin_{k}": v for k, v in log_c.items()},
                                            **{f"luna_{k}": v for k, v in log_l.items()}, "grad_norm": grad_norm, "skipped": skipped})
    save(run / "checkpoints" / "best.pt", model, recovery, step)
    shutil.copy(src / "p_min.json", run / "p_min.json")
    print(run)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor", required=True)
    args, rest = ap.parse_known_args()
    main(Path(args.extractor), rest)
