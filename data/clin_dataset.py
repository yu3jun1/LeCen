"""读取 slot 缓存与临床表。

临床变量的标准化与缺失处理只在训练集上拟合（训练集均值/标准差；缺失按训练集均值填补，缺失指示保留在表中）。
slot 按 selection_<sel_hash>.csv 选择后补齐到批内最大 K；空集合患者 mask 全 0、empty = 1。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.io import read_json, read_npz

LABEL_COLS = ["patient_id", "time_days", "event", "treatment", "split"]


def load_table(cfg):
    meta = Path(cfg.paths.meta)
    clin = pd.read_csv(meta / "clinical.csv", dtype={"patient_id": str})
    splits = pd.read_csv(meta / "splits_clin.csv", dtype={"patient_id": str})[["patient_id", "split"]]
    df = clin.merge(splits, on="patient_id")
    feats = [c for c in df.columns if c not in LABEL_COLS]
    tr = df.split == "train"
    mu, sd = df.loc[tr, feats].mean(), df.loc[tr, feats].std().replace(0, 1)
    x = ((df[feats] - mu) / sd).fillna(0.0)
    return df, x


def load_slots(extractor_run, patient_ids):
    s = read_json(Path(extractor_run) / "slots.json")
    sel = pd.read_csv(Path(s["slot_dir"]) / f"selection_{s['sel_hash']}.csv", dtype={"patient_id": str, "slots": str}).fillna({"slots": ""})
    sel = sel.set_index("patient_id").slots
    h, g = [], []
    for pid in patient_ids:
        z = read_npz(Path(s["slot_dir"]) / f"{pid}.npz")
        h.append(z["h"][[int(i) for i in sel[pid].split()]])
        g.append(z["pool_g"])
    k = max(1, max(len(x) for x in h))
    mask = np.array([[i < len(x) for i in range(k)] for x in h])
    hp = np.stack([np.pad(x, ((0, k - len(x)), (0, 0))) for x in h])
    return {"h": torch.tensor(hp, dtype=torch.float32), "mask": torch.tensor(mask), "pool_g": torch.tensor(np.stack(g)),
            "empty": torch.tensor(~mask.any(1), dtype=torch.float32)}


def cohort(cfg, df, x):
    """全部患者的张量：c、a（治疗类别下标）、time、event，以及 slot / pool_g。"""
    treat = {t: i for i, t in enumerate(sorted(df.treatment.unique()))}
    data = {"c": torch.tensor(x.to_numpy(), dtype=torch.float32), "a": torch.tensor(df.treatment.map(treat).to_numpy()),
            "time": df.time_days.to_numpy(np.float32), "event": df.event.to_numpy(np.float32)}
    data.update(load_slots(cfg.slots.extractor, df.patient_id))
    return data, len(treat)


def take(data, idx):
    return {k: v[idx] for k, v in data.items() if torch.is_tensor(v)}
