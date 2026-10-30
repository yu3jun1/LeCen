"""临床定位适配（仅在临床训练集与验证集上的肿瘤覆盖率不足时使用）。

python -m engine.adapt_clinical --extractor runs/D1/<run_id> adapt.iters=<步数>
torchrun --nproc_per_node=<GPU 数> -m engine.adapt_clinical --extractor runs/D1/<run_id> adapt.iters=<步数>

每个 optimizer 步混合一个临床训练批次与一个 LUNA16 训练批次（1:1）；D1、D2 使用完全相同的 LUNA16 批次（同一数据种子）。
临床批次（部分标注）：已标注肿瘤匹配后计算存在性正例、L1 与 GIoU；与某个已标注肿瘤 3D IoU > 0.3 的未匹配 query 作为负例；
其余未匹配 query 存在性权重为 0；热图只计已标注肿瘤中心的正例与尺寸（候选来自热图，覆盖不足首先要在这里修正）。LUNA16 批次：完整检测损失；D1 额外计算恢复损失（只在 LUNA16 批次上）。
AMP 与梯度裁剪顺序、DDP 方式同训练。适配后的权重存为新运行目录的 checkpoints/best.pt，提取 slot 时得到新的 extractor_hash。
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf

from common.dist import is_main, reduce_log, setup, teardown, world, wrap
from common.hashing import file_sha1
from common.io import ROOT, append_csv, load_config, make_run
from common.seed import set_determinism
from data.det_dataset import DetPatchDataset, loader, luna_dataset, to_device
from engine.train_det import TrainStep, build_optim, optimizer_step, save, save_failure
from models.detector import Detector
from models.recovery import RecoveryDecoder


def main(src, overrides):
    set_determinism()
    dev = setup()
    cfg = OmegaConf.merge(OmegaConf.load(src / "config.yaml"), OmegaConf.from_dotlist(overrides))
    paths = load_config(ROOT / "configs" / "data.yaml").paths
    # Keep pinned LUNA meta/processed paths; the current cache also exposes LUNA tokens.
    for key in ("clinical_meta", "clinical_processed", "cache"):
        cfg.paths[key] = paths[key]
    cfg.name, cfg.train.iters = f"{cfg.name}_adapt", cfg.adapt.iters
    ckpt = torch.load(src / "checkpoints" / "best.pt", map_location="cpu")
    run = make_run(cfg, {"source_run": str(src), "source_ckpt": file_sha1(src / "checkpoints" / "best.pt"),
                         "world_size": world()}) if is_main() else None
    model = Detector(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(dev)
    recovery = None
    if cfg.model.recovery.enabled:
        recovery = RecoveryDecoder(cfg.model.dim)
        recovery.load_state_dict(ckpt["recovery"])
        recovery.to(dev)
    patch = torch.tensor(list(cfg.patch_zyx), dtype=torch.float32, device=dev)
    step_fn = TrainStep(model, recovery, cfg, patch).train()
    ddp = wrap(step_fn, dev)
    params = list(step_fn.parameters())
    optimizer, scheduler = build_optim(cfg, model, list(recovery.parameters()) if recovery else [])
    p = Path(paths.clinical_processed)
    splits = pd.read_csv(Path(paths.clinical_meta) / "splits_clin.csv", dtype={"patient_id": str})
    clin = DetPatchDataset(cfg, splits[splits.split == "train"].patient_id, p / "clinical", None, p / "clinical_targets", True)
    clin_batches = loader(cfg, clin, cfg.seeds.data)
    luna_batches = loader(cfg, luna_dataset(cfg, "train", augment=True), cfg.seeds.data)
    step = 0
    while step < cfg.train.iters:
        bc, bl = to_device(next(clin_batches), dev), to_device(next(luna_batches), dev)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.train.amp):
                loss, [(_, log_c, _, _), (_, log_l, _, _)] = ddp([(bc, True), (bl, False)])
            loss.backward()
            grad_norm = optimizer_step(params, optimizer, scheduler, cfg.optim.clip_norm)
        except (ValueError, RuntimeError) as exc:
            save_failure(cfg, model, recovery, optimizer, scheduler, {"clinical": bc, "luna": bl}, step, exc)
            raise
        step += 1
        log = {**{f"clin_{k}": v for k, v in reduce_log(log_c, dev).items()},
               **{f"luna_{k}": v for k, v in reduce_log(log_l, dev).items()}}
        if is_main():
            append_csv(run / "train_log.csv", {"step": step, **log, "grad_norm": grad_norm})
    if is_main():
        save(run / "checkpoints" / "best.pt", model, recovery, step)
        shutil.copy(src / "p_min.json", run / "p_min.json")
        print(run)
    return run


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor", required=True)
    args, rest = ap.parse_known_args()
    main(Path(args.extractor), rest)
    teardown()
