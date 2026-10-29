"""检测训练（D1 / D2）。

python -m engine.train_det --cfg configs/det/D1.yaml                              # 单卡
torchrun --nproc_per_node=<GPU 数> -m engine.train_det --cfg configs/det/D1.yaml   # DDP

共享模块加载同一份 init.pt；D1 的恢复 decoder 在独立随机流（种子 1001）中初始化；数据采样使用独立 Generator（种子 2000）。
iters 指 optimizer 实际更新步数：AMP 下先反缩放再裁剪梯度；本步因 inf/NaN 被跳过时不推进 scheduler 与步数。
每 val_every 步在验证集上全胸推理（无增强）→ FROC → 保存 last.pt，CPM 最好时保存 best.pt（不存优化器状态）。
D1 在第一个含可恢复病灶的 batch 上做恢复梯度检查，写入 mech/grad_check.json。
DDP：train.batch 为全局 batch，按 rank 均分；验证集按 rank 分片推理后汇总到主进程；日志与检查点只由主进程写。
D1 与 D2 须使用相同的卡数。
"""
import argparse
import math
import os
from pathlib import Path
import time

import pandas as pd
import torch
import torch.distributed as dist
from torch import nn

from common.dist import gather_dict, is_main, rank, reduce_log, setup, teardown, world, wrap
from common.hashing import file_sha1
from common.io import append_csv, data_hashes, load_config, make_run, write_json
from common.seed import seeded, set_determinism
from data.det_dataset import loader, luna_dataset, to_device
from engine.infer_det import predict, preds_frame
from eval.froc import froc_metrics, scan_records
from eval.recall_chain import chain_summary
from losses.det_loss import det_losses
from losses.heatmap_loss import heatmap_loss
from losses.matching import match_layers
from losses.recovery_loss import recovery_term
from models.detector import Detector
from models.recovery import RecoveryDecoder


def build(cfg, dev):
    model = Detector(cfg)
    model.load_state_dict(torch.load(cfg.init_ckpt, map_location="cpu"))
    recovery = None
    if cfg.model.recovery.enabled:
        with seeded(cfg.seeds.recovery_init):
            recovery = RecoveryDecoder(cfg.model.dim).to(dev)
    return model.to(dev), recovery


def param_groups(model, extra, o):
    backbone = list(model.backbone.parameters())
    ids = {id(p) for p in backbone}
    rest = [p for p in model.parameters() if id(p) not in ids] + extra
    return [{"params": backbone, "lr": o.lr_backbone}, {"params": rest, "lr": o.lr_decoder}]


def warmup_cosine(o, iters):
    def f(step):
        if step < o.warmup_iters:
            return (step + 1) / o.warmup_iters
        return 0.5 * (1 + math.cos(math.pi * (step - o.warmup_iters) / (iters - o.warmup_iters)))
    return f


def build_optim(cfg, model, extra):
    o = cfg.optim
    optimizer = torch.optim.AdamW(param_groups(model, extra, o), weight_decay=o.wd)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(o, cfg.train.iters))
    return optimizer, scheduler, torch.amp.GradScaler("cuda", enabled=cfg.train.amp)


def optimizer_step(params, optimizer, scaler, scheduler, clip_norm):
    """先反缩放，再裁剪；返回 (是否实际更新, 裁剪前梯度范数)。"""
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=clip_norm)
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    stepped = scaler.get_scale() >= scale_before
    if stepped:
        scheduler.step()
    return stepped, float(grad_norm)


def compute_loss(model, recovery, batch, cfg, patch, partial=False):
    """L = w_hm·L_hm + w_size·L_size + Σ_layers(…) + w_local·L_local。部分标注批次（临床）的热图只计正例与尺寸。"""
    w = cfg.loss.w
    out = model(batch["image"], batch["tokens"], batch["token_xyz"], batch)
    matches = match_layers(out, batch["boxes"], cfg, patch)
    l_det, log = det_losses(out, batch, matches, cfg, patch, partial)
    l_hm, l_size = heatmap_loss(out["hm"], out["size"], batch, cfg.model.heatmap.level, partial)
    l_det = l_det + w.hm * l_hm + w.size * l_size
    log.update(hm=l_hm, size=l_size)
    loss, terms = l_det, {"det": l_det}
    if recovery is not None:
        l_local, n_used, n_zero = recovery_term(recovery, out, batch, matches[-1], cfg, patch)
        loss, terms["local"] = loss + w.local * l_local, l_local
        log.update(local=l_local, n_rec=n_used, n_rec_zero=n_zero)
    return loss, {k: torch.as_tensor(v).item() for k, v in log.items()}, out, terms


class TrainStep(nn.Module):
    """检测器、恢复 decoder 与损失放在同一个 forward 中，供 DDP 包装。

    多卡时把 0·Σp 加入损失（ddp.zero_loss_for_unused）：某个 rank 某步跳过恢复 decoder、没有 GT
    （size head 与框回归 head 不进入损失）时，这些参数仍在计算图中，各 rank 的梯度同步保持一致。
    """

    def __init__(self, model, recovery, cfg, patch):
        super().__init__()
        self.model, self.recovery, self.cfg, self.patch = model, recovery, cfg, patch
        self.zero_loss = dist.is_initialized() and cfg.ddp.zero_loss_for_unused

    def forward(self, batches):
        """batches: [(batch, partial)]；恢复损失只在完整标注（非 partial）批次上计算。"""
        res = [compute_loss(self.model, None if partial else self.recovery, b, self.cfg, self.patch, partial)
               for b, partial in batches]
        loss = sum(r[0] for r in res)
        if self.zero_loss:
            loss = loss + 0 * sum(p.sum() for p in self.parameters())
        return loss, res


def grad_check(model, out, terms, scaler):
    """∂L_local/∂h 非零；decoder 参数梯度在只用 L_det、只用 L_local 两种情况下分别计算并比较。"""
    h = out["layers"][-1]["h"]
    params = [p for p in model.decoder.parameters() if p.requires_grad]

    def grads(loss):
        g = torch.autograd.grad(scaler.scale(loss), params, retain_graph=True, allow_unused=True)
        return torch.cat([(x if x is not None else torch.zeros_like(p)).flatten().float() for x, p in zip(g, params)])

    g_h = torch.autograd.grad(scaler.scale(terms["local"]), h, retain_graph=True)[0]
    g_det, g_local = grads(terms["det"]), grads(terms["local"])
    return {"dL_local_dh_norm": float(g_h.float().norm()), "decoder_grad_norm_det_only": float(g_det.norm()),
            "decoder_grad_norm_local_only": float(g_local.norm()),
            "cosine_det_local": float(torch.nn.functional.cosine_similarity(g_det, g_local, 0)),
            "loss_scale": scaler.get_scale()}


def validate(model, cfg, ids, dev):
    """各 rank 推理验证集的一个分片，汇总后由主进程计算指标（其余 rank 返回 None）。"""
    stages = predict(model, cfg, list(ids)[rank()::world()], Path(cfg.paths.processed) / "luna16", dev)
    stages = gather_dict({sid: {k: v for k, v in s.items() if k != "dec_h"} for sid, s in stages.items()})
    model.train()
    if not is_main():
        return None
    rec = scan_records(cfg, preds_frame(stages, cfg.infer.p_min), list(stages))
    m = froc_metrics(rec)
    return {"CPM": m["cpm"], "sens@1FP": m["sens@1"], "sens@1FP_3-6mm": m["sens1fp_3to6"], **chain_summary(cfg, stages, rec)}


def save(path, model, recovery, step):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "recovery": recovery.state_dict() if recovery else None, "step": step}, path)


def save_failure(cfg, model, recovery, optimizer, scheduler, scaler, batch, step, skipped, error):
    """Opt-in diagnostics for a failed forward; no automatic training resume."""
    directory = cfg.train.get("failure_dir", None)
    if directory is None:
        return
    path = Path(directory) / f"{cfg.name}_step{step}_rank{rank()}_{os.getpid()}_{time.time_ns()}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "recovery": recovery.state_dict() if recovery else None,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "batch": batch, "step": step, "skipped": skipped,
                "error": str(error), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None}, path)
    print(f"Saved failure diagnostic: {path}", flush=True)


def train(cfg):
    set_determinism()
    dev = setup()
    run = make_run(cfg, {"init_ckpt": file_sha1(cfg.init_ckpt), "world_size": world(), **data_hashes(cfg)}) if is_main() else None
    model, recovery = build(cfg, dev)
    patch = torch.tensor(list(cfg.patch_zyx), dtype=torch.float32, device=dev)
    step_fn = TrainStep(model, recovery, cfg, patch).train()
    ddp = wrap(step_fn, dev)
    params = list(step_fn.parameters())
    optimizer, scheduler, scaler = build_optim(cfg, model, list(recovery.parameters()) if recovery else [])
    batches = loader(cfg, luna_dataset(cfg, "train", augment=True), cfg.seeds.data)
    splits = pd.read_csv(Path(cfg.paths.meta) / "splits_det.csv")
    val_ids = splits[splits.split == "val"].seriesuid
    step, skipped, best, checked = 0, 0, -1.0, recovery is None or not is_main()
    while step < cfg.train.iters:
        logs = []
        for micro in range(cfg.train.accum_steps):
            batch = to_device(next(batches), dev)
            try:
                with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.train.amp):
                    loss, [(_, log, out, terms)] = ddp([(batch, False)])
            except (ValueError, RuntimeError) as exc:
                save_failure(cfg, model, recovery, optimizer, scheduler, scaler, batch, step, skipped, exc)
                raise
            if not checked and log["n_rec"] > 0:
                write_json(run / "mech" / "grad_check.json", {"step": step, **grad_check(model, out, terms, scaler)})
                checked = True
            scaler.scale(loss / cfg.train.accum_steps).backward()
            logs.append(log)
        stepped, grad_norm = optimizer_step(params, optimizer, scaler, scheduler, cfg.optim.clip_norm)
        if not stepped:
            skipped += 1
            continue
        step += 1
        mean = reduce_log(pd.DataFrame(logs).mean().to_dict(), dev)
        if is_main():
            append_csv(run / "train_log.csv", {"step": step, **mean, "lr": scheduler.get_last_lr()[1],
                                                "grad_norm": grad_norm, "skipped": skipped})
        if step % cfg.train.val_every == 0:
            m = validate(model, cfg, val_ids, dev)
            if is_main():
                append_csv(run / "val_metrics.csv", {"step": step, **m})
                save(run / "checkpoints" / "last.pt", model, recovery, step)
                if m["CPM"] > best:
                    best = m["CPM"]
                    save(run / "checkpoints" / "best.pt", model, recovery, step)
    return run


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    args, rest = ap.parse_known_args()
    train(load_config(args.cfg, rest))
    teardown()
