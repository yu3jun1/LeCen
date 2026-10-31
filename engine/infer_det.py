"""全胸推理与后处理。

1. 预处理体积上做滑窗（窗口 = patch，重叠 25%，无增强），每个窗口取最终层全部 query 的 (p, box, h)。
2. 框换算到预处理体积坐标与世界坐标；丢弃中心落在体积外的框。
3. 中心距离 NMS：按 p 降序；候选中心落在已保留框的半径（0.5 × 最大边长，mm）内则抑制；框与 h 一起保留或删除。
4. 保留 p > p_min；验证集 FROC 未达到 8 FP/scan 时降低 p_min 并记录在 p_min.json（测试集沿用）。
5. 输出 preds_<split>.csv：seriesuid, coordX, coordY, coordZ, probability（官方评测格式）。

python -m engine.infer_det --run runs/D1/<run_id> --split val
python -m engine.infer_det --run runs/D1/<run_id> --split test --final
python -m engine.infer_det --run runs/D0a/<run_id> --split test --final --monai-json <MONAI 推理输出> --monai-config <MONAI 推理配置>
（D0a：先核对推理配置实际加载的权重与 <run>/checkpoints/model.pt 的 SHA256 一致，再转换）
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from common.boxes import patch_box_to_world
from common.coords import patch_to_vol, patch_to_world
from common.hashing import file_sha1
from common.io import ROOT, load_config, read_json, register_final, write_csv, write_json
from data.det_dataset import to_device
from data.infer_dataset import InferCase
from eval.froc import scan_records
from models.detector import Detector


@torch.no_grad()
def infer_case(model, cfg, case, dev):
    """返回各阶段输出（世界坐标）：hm_xyz 热图候选中心；dec_* 最终层 query（中心在体积内）；keep 为 NMS 保留的 dec 下标。
    infer.window_core 时，热图候选与 decoder 输出只保留中心落在所在窗口中心区域内的部分（每个位置只由一个窗口预测）。"""
    patch = torch.tensor(list(cfg.patch_zyx), dtype=torch.float32, device=dev)
    shape = torch.tensor(case.shape, dtype=torch.float32, device=dev)

    def kept(vol, b):
        k = ((vol >= -0.5) & (vol <= shape - 0.5)).all(-1)
        if cfg.infer.window_core:
            k &= ((vol >= b["core_lo"][:, None]) & (vol < b["core_hi"][:, None])).all(-1)
        return k
    hm_xyz, boxes, ps, hs = [], [], [], []
    for i in range(0, len(case.starts), cfg.train.batch):
        b = to_device(case.batch(case.starts[i:i + cfg.train.batch]), dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.train.amp):
            out = model(b["image"], b["tokens"], b["token_xyz"], b)
        last = out["layers"][-1]
        off, flip = b["offset_vox"][:, None], b["flip"][:, None]
        vol_c = patch_to_vol(last["box"][..., :3], off, flip, patch)
        keep = kept(vol_c, b)
        boxes.append(patch_box_to_world(last["box"], off, flip, patch, b["affine"], b["spacing"][:, None])[keep])
        ps.append(last["exist_logit"].sigmoid()[keep])
        hs.append(last["h"].float()[keep])
        ref_vol = patch_to_vol(out["ref_center"], off, flip, patch)
        ref_in = kept(ref_vol, b)
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


def score_threshold_mask(scores, threshold, *, inclusive=False):
    """Compare in float64 so NumPy cannot round a scalar threshold to float32.

    Strict p > p_min filtering and inclusive FROC thresholds use the same
    precision, including repeated scores originating from BF16 inference.
    """
    scores = np.asarray(scores, dtype=np.float64)
    threshold = np.float64(threshold)
    return scores >= threshold if inclusive else scores > threshold


def preds_frame(stages, p_min):
    rows = []
    for sid, s in stages.items():
        k = s["keep"][score_threshold_mask(s["dec_p"][s["keep"]], p_min)]
        rows += [(sid, *s["dec_box"][i, :3], s["dec_p"][i]) for i in k]
    return pd.DataFrame(rows, columns=["seriesuid", "coordX", "coordY", "coordZ", "probability"])


def fit_p_min(stages, cfg, p_min, target_fp=8.0):
    """验证集：若 p > p_min 达不到 target_fp FP/scan，降低阈值以包含目标分数及其全部同分候选（候选不足时为 0）。"""
    rec = scan_records(cfg, preds_frame(stages, 0.0), list(stages))
    fp = np.sort(np.asarray(np.concatenate([r["fp"] for r in rec.values()]), dtype=np.float64))[::-1]
    n = int(np.ceil(target_fp * len(stages)))
    if score_threshold_mask(fp, p_min).sum() >= n:
        return p_min
    # The strict filter must include the target score and every tied candidate.
    return float(np.nextafter(fp[n - 1], -np.inf)) if len(fp) >= n else 0.0


def monai_preds(path):
    """MONAI bundle 推理输出（世界坐标 cccwhd 与 label_scores）→ 官方 csv 格式。"""
    rows = []
    for items in read_json(path).values():
        for it in items:
            sid = Path(it["image"]).name.removesuffix(".nii.gz")
            rows += [(sid, *b[:3], s) for b, s in zip(it["box"], it["label_scores"])]
    return pd.DataFrame(rows, columns=["seriesuid", "coordX", "coordY", "coordZ", "probability"])


def monai_checkpoint(config_path):
    """MONAI 推理配置实际加载的权重：(load_pretrain, checkpointloader#load_path)，按 bundle 配置表达式解析。"""
    from monai.bundle import ConfigParser

    parser = ConfigParser()
    parser.read_config(str(config_path))
    return bool(parser.get_parsed_content("load_pretrain")), Path(parser.get_parsed_content("checkpointloader#load_path"))


def verify_monai_checkpoint(run, config_path, split):
    """D0a 预测必须来自该运行验证集选出的权重 <run>/checkpoints/model.pt：推理配置须启用加载，
    且加载文件与之 SHA256 相同；否则拒绝转换。通过时把两者路径与哈希写入 <run>/monai_<split>_provenance.json。"""
    expected = Path(run) / "checkpoints" / "model.pt"
    load, loaded = monai_checkpoint(config_path)
    record = {"monai_config": str(config_path), "load_pretrain": load,
              "loaded": str(loaded), "loaded_sha256": file_sha1(loaded, "sha256") if loaded.is_file() else None,
              "expected": str(expected), "expected_sha256": file_sha1(expected, "sha256")}
    if not load or record["loaded_sha256"] != record["expected_sha256"]:
        raise ValueError(f"D0a 推理没有加载该运行的权重：{record}")
    write_json(Path(run) / f"monai_{split}_provenance.json", record)
    return record


def load_detector(run, dev, ckpt="best.pt"):
    cfg = OmegaConf.load(Path(run) / "config.yaml")
    model = Detector(cfg)
    model.load_state_dict(torch.load(Path(run) / "checkpoints" / ckpt, map_location="cpu")["model"])
    return cfg, model.to(dev).eval()


def main(run, split, final, monai_json, monai_config):
    run = Path(run)
    if monai_json:
        verify_monai_checkpoint(run, monai_config, split)
        OmegaConf.save(load_config(ROOT / "configs" / "det" / "D0a_monai.yaml"), run / "config.yaml", resolve=True)
    cfg = OmegaConf.load(run / "config.yaml")
    if final:
        register_final(cfg.name, run, run / "checkpoints" / ("model.pt" if monai_json else "best.pt"))
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
    ap.add_argument("--monai-config", help="生成 --monai-json 时 MONAI 实际使用的推理配置；用于核对加载的权重")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        ap.error("测试集评测需显式传入 --final")
    if bool(args.monai_json) != bool(args.monai_config):
        ap.error("--monai-json 与 --monai-config 必须同时给出")
    main(args.run, args.split, args.final, args.monai_json, args.monai_config)
