"""全胸推理与后处理。

1. 预处理体积上做滑窗（窗口 = patch，重叠 25%，无增强），每个窗口取最终层全部 query 的 (p, box, h)。
2. 框换算到预处理体积坐标与世界坐标；丢弃中心落在体积外的框。
3. 中心距离 NMS：按 p 降序；候选中心落在已保留框的半径（0.5 × 最大边长，mm）内则抑制；框与 h 一起保留或删除。
4. 保留 p > p_min；验证集 FROC 未达到 8 FP/scan 时降低 p_min 并记录在 p_min.json（测试集沿用）。
5. 输出 preds_<split>.csv：seriesuid, coordX, coordY, coordZ, probability（官方评测格式）。

python -m engine.infer_det --run runs/D1/<run_id> --split val
python -m engine.infer_det --run runs/D1/<run_id> --split test --final
python -m engine.infer_det --run runs/D0a/<run_id> --split test --final --monai-json <MONAI 推理输出>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from common.boxes import patch_box_to_world
from common.coords import patch_to_vol, patch_to_world
from common.io import ROOT, load_config, read_json, register_final, write_csv, write_json
from data.det_dataset import to_device
from data.infer_dataset import InferCase
from eval.froc import scan_records
from models.detector import Detector


@torch.no_grad()
def infer_case(model, cfg, case, dev):
    """返回各阶段输出（世界坐标）：hm_xyz 热图候选中心；dec_* 最终层 query（中心在体积内）；keep 为 NMS 保留的 dec 下标。"""
    patch = torch.tensor(list(cfg.patch_zyx), dtype=torch.float32, device=dev)
    shape = torch.tensor(case.shape, dtype=torch.float32, device=dev)
    hm_xyz, boxes, ps, hs = [], [], [], []
    for i in range(0, len(case.starts), cfg.train.batch):
        b = to_device(case.batch(case.starts[i:i + cfg.train.batch]), dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.train.amp):
            out = model(b["image"], b["tokens"], b["token_xyz"], b)
        last = out["layers"][-1]
        off, flip = b["offset_vox"][:, None], b["flip"][:, None]
        vol_c = patch_to_vol(last["box"][..., :3], off, flip, patch)
        keep = ((vol_c >= -0.5) & (vol_c <= shape - 0.5)).all(-1)
        boxes.append(patch_box_to_world(last["box"], off, flip, patch, b["affine"], b["spacing"][:, None])[keep])
        ps.append(last["exist_logit"].sigmoid()[keep])
        hs.append(last["h"].float()[keep])
        ref_vol = patch_to_vol(out["ref_center"], off, flip, patch)
        ref_in = ((ref_vol >= -0.5) & (ref_vol <= shape - 0.5)).all(-1)
        hm_xyz.append(patch_to_world(out["ref_center"], off, flip, patch, b["affine"])[ref_in])
    boxes, p = torch.cat(boxes).cpu().numpy(), torch.cat(ps).cpu().numpy()
    return {"hm_xyz": torch.cat(hm_xyz).cpu().numpy(), "dec_box": boxes, "dec_p": p, "dec_h": torch.cat(hs).cpu().numpy(),
            "keep": center_nms(boxes, p, cfg.infer.nms_radius_scale)}


def center_nms(boxes, p, radius_scale):
    keep = []
    for i in np.argsort(-p, kind="stable"):
        k = np.array(keep, dtype=int)
        r = radius_scale * boxes[k, 3:].max(1)
        if not (np.linalg.norm(boxes[k, :3] - boxes[i, :3], axis=1) < r).any():
            keep.append(i)
    return np.array(keep, dtype=int)


def predict(model, cfg, ids, vol_dir, dev):
    model.eval()
    token_dir = Path(cfg.paths.cache) / "ctclip"
    return {cid: infer_case(model, cfg, InferCase(cid, vol_dir, token_dir, cfg.patch_zyx, cfg.infer.overlap), dev) for cid in ids}


def preds_frame(stages, p_min):
    rows = []
    for sid, s in stages.items():
        k = s["keep"][s["dec_p"][s["keep"]] > p_min]
        rows += [(sid, *s["dec_box"][i, :3], s["dec_p"][i]) for i in k]
    return pd.DataFrame(rows, columns=["seriesuid", "coordX", "coordY", "coordZ", "probability"])


def fit_p_min(stages, cfg, p_min, target_fp=8.0):
    """验证集：p > p_min 的候选若达不到 target_fp FP/scan，把 p_min 降到恰好达到的位置（候选不足时为 0）。"""
    rec = scan_records(cfg, preds_frame(stages, 0.0), list(stages))
    fp = np.sort(np.concatenate([r["fp"] for r in rec.values()]))[::-1]
    n = int(np.ceil(target_fp * len(stages)))
    if (fp > p_min).sum() >= n:
        return p_min
    return float(np.nextafter(fp[n - 1], 0)) if len(fp) >= n else 0.0


def monai_preds(path):
    """MONAI bundle 推理输出（世界坐标 cccwhd 与 label_scores）→ 官方 csv 格式。"""
    rows = []
    for items in read_json(path).values():
        for it in items:
            sid = Path(it["image"]).name.removesuffix(".nii.gz")
            rows += [(sid, *b[:3], s) for b, s in zip(it["box"], it["label_scores"])]
    return pd.DataFrame(rows, columns=["seriesuid", "coordX", "coordY", "coordZ", "probability"])


def load_detector(run, dev, ckpt="best.pt"):
    cfg = OmegaConf.load(Path(run) / "config.yaml")
    model = Detector(cfg)
    model.load_state_dict(torch.load(Path(run) / "checkpoints" / ckpt, map_location="cpu")["model"])
    return cfg, model.to(dev).eval()


def main(run, split, final, monai_json):
    run = Path(run)
    if monai_json:
        run.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(load_config(ROOT / "configs" / "det" / "D0a_monai.yaml"), run / "config.yaml", resolve=True)
    cfg = OmegaConf.load(run / "config.yaml")
    if final:
        register_final(cfg.name, run, monai_json or run / "checkpoints" / "best.pt")
    if monai_json:
        write_csv(run / f"preds_{split}.csv", monai_preds(monai_json))
        return
    cfg, model = load_detector(run, "cuda")
    splits = pd.read_csv(Path(cfg.paths.meta) / "splits_det.csv")
    stages = predict(model, cfg, splits[splits.split == split].seriesuid, Path(cfg.paths.processed) / "luna16", "cuda")
    if split == "val":
        write_json(run / "p_min.json", {"p_min": fit_p_min(stages, cfg, cfg.infer.p_min), "config_p_min": cfg.infer.p_min})
    write_csv(run / f"preds_{split}.csv", preds_frame(stages, read_json(run / "p_min.json")["p_min"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--monai-json")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        ap.error("测试集评测需显式传入 --final")
    main(args.run, args.split, args.final, args.monai_json)
