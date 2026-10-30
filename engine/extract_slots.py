"""在临床 CT 上提取并缓存 slot，并确定病灶选择。

每位患者：全胸推理 → NMS 后 p > p_min 的候选 → cache/slots/<extractor_hash>/<patient_id>.npz：
h (K×256)、p (K)、box_world (K×6)、raw_count 与元数据
（含输入 series_uid、预处理 CT 与 CT-CLIP token 文件的 sha1）。已有同名缓存而元数据不一致时拒绝写入。
选择：在临床验证集上，以主肿瘤覆盖率（被选 slot 中心落入肿瘤包围框外扩 cover_margin_mm）为目标，
取能达到最大覆盖率的最小 K（≤ k_max），再取该 K 下仍达到最大覆盖率的最大 p 阈值；测试集沿用。
验证集没有已标注肿瘤或覆盖率为 0 时停止：此时应先做临床定位适配（engine/adapt_clinical.py）。
结果写入 selection_<sel_hash>.csv（patient_id、被选 slot 下标、阈值、K_max、是否空集合）。

python -m engine.extract_slots --extractor runs/D1/<run_id>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from common.hashing import extractor_hash, file_sha1, sel_hash
from common.io import ROOT, load_config, read_json, read_npz, world_list, write_csv, write_json, write_npz
from data.infer_dataset import InferCase
from engine.infer_det import infer_case, load_detector, score_threshold_mask


def extract(run, dev):
    cfg, model = load_detector(run, dev)
    p_min = read_json(run / "p_min.json")["p_min"]
    ehash = extractor_hash(cfg, run / "checkpoints" / "best.pt", p_min)
    out_dir = Path(cfg.paths.cache) / "slots" / ehash
    vol_dir, token_dir = Path(cfg.paths.processed) / "clinical", Path(cfg.paths.cache) / "ctclip"
    splits = pd.read_csv(Path(cfg.paths.meta) / "splits_clin.csv", dtype={"patient_id": str})
    series = pd.read_csv(Path(cfg.paths.meta) / "clinical.csv", dtype={"patient_id": str}).set_index("patient_id").series_uid
    for r in splits.itertuples():
        meta = {"patient_id": r.patient_id, "split": r.split, "extractor_hash": ehash, "preprocessing_version": cfg.preprocess.version,
                "series_uid": series[r.patient_id], "ct_sha1": file_sha1(vol_dir / f"{r.patient_id}.nii.gz"),
                "tokens_sha1": file_sha1(token_dir / f"{r.patient_id}.npz")}
        path = out_dir / f"{r.patient_id}.npz"
        if path.exists():
            old = read_npz(path)
            if any(str(old[k]) != str(v) for k, v in meta.items()):
                raise RuntimeError(f"{path} 已存在且元数据不一致：{ {k: str(old[k]) for k in meta} } ≠ {meta}")
            continue
        case = InferCase(r.patient_id, vol_dir, token_dir, cfg.patch_zyx, cfg.infer.overlap)
        st = infer_case(model, cfg, case, dev)
        k = st["keep"][score_threshold_mask(st["dec_p"][st["keep"]], p_min)]
        write_npz(path, h=st["dec_h"][k], p=st["dec_p"][k], box_world=st["dec_box"][k],
                  raw_count=np.array(len(k)), **{k_: np.array(v) for k_, v in meta.items()})
    return cfg, ehash, out_dir


def covered(box_world, tumor, margin):
    return (np.abs(box_world[:, :3] - tumor[:3]) <= tumor[3:] / 2 + margin).all(1)


def select(cfg, sel, ehash, out_dir):
    meta = Path(cfg.paths.meta)
    splits = pd.read_csv(meta / "splits_clin.csv", dtype={"patient_id": str})
    tumors = pd.read_csv(meta / "clinical_tumors.csv", dtype={"patient_id": str})
    slots = {pid: read_npz(out_dir / f"{pid}.npz") for pid in splits.patient_id}
    val = []
    for pid in splits[splits.split == "val"].patient_id:
        t = world_list(tumors[tumors.patient_id == pid].bbox_world)
        if len(t):
            s = slots[pid]
            order = np.argsort(-s["p"], kind="stable")
            val.append((s["p"][order], covered(s["box_world"][order], t[np.argmax(t[:, 3:].prod(1))], sel.cover_margin_mm)))

    def coverage(thr, k):
        return np.mean([(c[:k] & (p[:k] >= thr)).any() for p, c in val])

    if not val:
        raise RuntimeError("临床验证集没有已标注肿瘤，无法确定 slot 选择")
    target = coverage(0.0, sel.k_max)
    if target == 0:
        raise RuntimeError("临床验证集主肿瘤覆盖率为 0：先做临床定位适配（engine/adapt_clinical.py）后重新提取")
    k_max = next(k for k in range(1, sel.k_max + 1) if coverage(0.0, k) == target)
    thr = max(t for t in np.unique(np.concatenate([p for p, _ in val])) if coverage(t, k_max) == target)
    rows = []
    for pid, s in slots.items():
        idx = np.argsort(-s["p"], kind="stable")[:k_max]
        idx = idx[s["p"][idx] >= thr]
        rows.append({"patient_id": pid, "slots": " ".join(map(str, idx)), "threshold": thr, "k_max": k_max, "is_empty": len(idx) == 0})
    df = pd.DataFrame(rows)
    shash = sel_hash(ehash, sel)
    write_csv(out_dir / f"selection_{shash}.csv", df)
    print(f"val 主肿瘤覆盖率 {target:.3f}；K_max={k_max}，阈值={thr:.4f}；空集合比例 {df.is_empty.mean():.3f}")
    return shash, {"val_coverage": float(target), "empty_ratio": float(df.is_empty.mean())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor", required=True)
    run = Path(ap.parse_args().extractor)
    cfg, ehash, out_dir = extract(run, "cuda")
    sel = OmegaConf.masked_copy(load_config(ROOT / "configs" / "clin" / "base.yaml").slots, ["k_max", "cover_margin_mm"])
    shash, stats = select(cfg, sel, ehash, out_dir)
    write_json(run / "slots.json", {"extractor_hash": ehash, "sel_hash": shash, "slot_dir": str(out_dir), **stats})
