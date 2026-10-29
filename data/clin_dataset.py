"""读取临床表、全局特征（G0）与 slot 缓存。

临床变量的标准化与缺失处理只在训练集上拟合：只保留训练集上方差非零的列（列表随之冻结用于验证、测试），
按训练集均值/标准差标准化，缺失按训练集均值填补，缺失指示保留在表中。
全局特征 g_global（G0）：缓存的 12×12×12 CT-CLIP token 在空间维度上的均值（512 维），与检测器无关，
所有影像生存配置共用；按训练集逐维标准化。它是 token 均值池化表示，不是 CT-CLIP 官方的图像级 latent。
slot 只在 model.slots 时读取，按 selection_<sel_hash>.csv 选择后补齐到批内最大 K；空集合患者 mask 全 0、empty = 1。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.hashing import files_sha1
from common.io import read_json, read_npz

GLOBAL_FEATURE = "mean of cached 12x12x12 CT-CLIP tokens (1728x512 -> 512), train-set z-score"

LABEL_COLS = ["patient_id", "series_uid", "time_days", "event", "treatment", "split"]


def load_table(cfg):
    meta = Path(cfg.paths.meta)
    clin = pd.read_csv(meta / "clinical.csv", dtype={"patient_id": str})
    splits = pd.read_csv(meta / "splits_clin.csv", dtype={"patient_id": str})[["patient_id", "split"]]
    df = clin.merge(splits, on="patient_id")
    tr = df.split == "train"
    feats = [c for c in df.columns if c not in LABEL_COLS and df.loc[tr, c].std() > 0]
    mu, sd = df.loc[tr, feats].mean(), df.loc[tr, feats].std()
    x = ((df[feats] - mu) / sd).fillna(0.0)
    return df, x


def token_paths(cfg, patient_ids):
    return [Path(cfg.paths.cache) / "ctclip" / f"{pid}.npz" for pid in patient_ids]


def load_global(cfg, patient_ids, train):
    """G0：tokens (1728, 512) 沿空间维度取均值 → (n, 512)，再用训练集均值/标准差逐维标准化。"""
    g = np.stack([read_npz(p)["tokens"].astype(np.float32).mean(axis=0) for p in token_paths(cfg, patient_ids)])
    return torch.tensor((g - g[train].mean(0)) / g[train].std(0), dtype=torch.float32)


def global_hashes(cfg, patient_ids):
    """G0 的来源与定义：CT-CLIP 权重、取 token 的层、池化网格、预处理版本，以及各患者 token 文件的组合哈希。"""
    return {"global_feature": GLOBAL_FEATURE, "ctclip_checkpoint": str(cfg.ctclip.checkpoint),
            "ctclip_token_layer": str(cfg.ctclip.token_layer), "ctclip_pooled_grid": list(cfg.ctclip.pooled_grid),
            "preprocessing_version": cfg.preprocess.version,
            "ctclip_tokens_sha1": files_sha1(token_paths(cfg, patient_ids))}


def load_slots(extractor_run, patient_ids):
    """病灶向量 h、有效位置 mask 与空集合标记 empty。"""
    s = read_json(Path(extractor_run) / "slots.json")
    sel = pd.read_csv(Path(s["slot_dir"]) / f"selection_{s['sel_hash']}.csv", dtype={"patient_id": str, "slots": str}).fillna({"slots": ""})
    sel = sel.set_index("patient_id").slots
    h = [read_npz(Path(s["slot_dir"]) / f"{pid}.npz")["h"][[int(i) for i in sel[pid].split()]] for pid in patient_ids]
    k = max(1, max(len(x) for x in h))
    mask = np.array([[i < len(x) for i in range(k)] for x in h])
    hp = np.stack([np.pad(x, ((0, k - len(x)), (0, 0))) for x in h])
    return {"h": torch.tensor(hp, dtype=torch.float32), "mask": torch.tensor(mask),
            "empty": torch.tensor(~mask.any(1), dtype=torch.float32)}


def cohort(cfg, df, x):
    """全部患者的张量：c、a（治疗类别下标）、time、event、g_global；model.slots 时另加 h、mask、empty。"""
    treat = {t: i for i, t in enumerate(sorted(df.treatment.unique()))}
    data = {"c": torch.tensor(x.to_numpy(), dtype=torch.float32), "a": torch.tensor(df.treatment.map(treat).to_numpy()),
            "time": df.time_days.to_numpy(np.float32), "event": df.event.to_numpy(np.float32),
            "g_global": load_global(cfg, df.patient_id, (df.split == "train").to_numpy())}
    if cfg.model.slots:
        data.update(load_slots(cfg.slots.extractor, df.patient_id))
    return data, len(treat)


def take(data, idx):
    return {k: v[idx] for k, v in data.items() if torch.is_tensor(v)}
