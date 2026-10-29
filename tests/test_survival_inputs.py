"""生存实验的输入可归因性：S1（G0）不依赖检测器与 slot；S1–S5 共用同一个 g_global，S4/S5 只差 slot 来源。"""
import numpy as np
import pandas as pd
import pytest
import torch

import data.clin_dataset as clin_dataset
from common.io import ROOT, load_config, write_json, write_npz
from data.clin_dataset import cohort, load_table
from engine.train_surv import fit_neural

N = 40


def make_cohort(tmp_path):
    rng = np.random.default_rng(0)
    pids = [f"P{i:02d}" for i in range(N)]
    meta, tok = tmp_path / "meta", tmp_path / "cache" / "ctclip"
    meta.mkdir()
    tok.mkdir(parents=True)
    pd.DataFrame({"patient_id": pids, "series_uid": pids, "time_days": rng.uniform(30, 1500, N),
                  "event": (rng.random(N) < 0.6).astype(int), "treatment": ["A", "B"] * (N // 2),
                  "age": rng.normal(60, 8, N)}).to_csv(meta / "clinical.csv", index=False)
    pd.DataFrame({"patient_id": pids, "split": ["train"] * 24 + ["val"] * 8 + ["test"] * 8}).to_csv(meta / "splits_clin.csv", index=False)
    for pid in pids:
        write_npz(tok / f"{pid}.npz", tokens=rng.standard_normal((1728, 512)).astype(np.float16))
    return pids, [f"paths.meta={meta}", f"paths.cache={tmp_path / 'cache'}", "train.max_epochs=3"]


def make_extractor(tmp_path, name, pids, scale):
    """模拟一个提取器运行目录：slots.json、选择表与每位患者的 slot 缓存（256 维）。"""
    rng = np.random.default_rng(scale)
    run, slot_dir = tmp_path / name, tmp_path / f"slots_{name}"
    slot_dir.mkdir()
    write_json(run / "slots.json", {"slot_dir": str(slot_dir), "sel_hash": "s", "extractor_hash": name})
    pd.DataFrame({"patient_id": pids, "slots": ["0 1"] * len(pids)}).to_csv(slot_dir / "selection_s.csv", index=False)
    for pid in pids:
        write_npz(slot_dir / f"{pid}.npz", h=(scale * rng.standard_normal((3, 256))).astype(np.float32))
    return run


def test_s1_runs_without_detector_or_slots(tmp_path, monkeypatch):
    _, ov = make_cohort(tmp_path)

    def forbidden(*args):
        raise AssertionError("S1 不应读取 slot")

    monkeypatch.setattr(clin_dataset, "load_slots", forbidden)
    cfg = load_config(ROOT / "configs" / "clin" / "S1.yaml", ov)
    assert cfg.slots.extractor is None and not cfg.model.slots
    df, x = load_table(cfg)
    data, _ = cohort(cfg, df, x)
    assert "h" not in data and data["g_global"].shape == (N, 512)
    run = tmp_path / "run"
    run.mkdir()
    surv = fit_neural(cfg, df, x, np.array([100.0, 300.0]), run)((df.split == "test").to_numpy())
    assert surv.shape == (8, 2) and np.isfinite(surv).all()


def test_global_shared_across_configs(tmp_path):
    pids, ov = make_cohort(tmp_path)
    d1, d2 = make_extractor(tmp_path, "D1", pids, 1), make_extractor(tmp_path, "D2", pids, 2)
    data = {}
    for name, ext in (("S1", None), ("S2", d1), ("S3", d1), ("S4", d2), ("S5", d1)):
        cfg = load_config(ROOT / "configs" / "clin" / f"{name}.yaml", ov + ([f"slots.extractor={ext}"] if ext else []))
        df, x = load_table(cfg)
        data[name] = cohort(cfg, df, x)[0]
    for name in ("S2", "S3", "S4", "S5"):
        assert torch.equal(data[name]["g_global"], data["S1"]["g_global"])
        assert data[name]["h"].shape[2] == 256
    assert not torch.equal(data["S4"]["h"], data["S5"]["h"])
    assert torch.equal(data["S2"]["h"], data["S5"]["h"])


@pytest.mark.parametrize("name", ["S2", "S5"])
def test_model_dims(tmp_path, name):
    pids, ov = make_cohort(tmp_path)
    cfg = load_config(ROOT / "configs" / "clin" / f"{name}.yaml", ov + [f"slots.extractor={make_extractor(tmp_path, 'D1', pids, 1)}"])
    df, x = load_table(cfg)
    run = tmp_path / "run"
    run.mkdir()
    surv = fit_neural(cfg, df, x, np.array([100.0, 300.0]), run)((df.split == "test").to_numpy())
    assert np.isfinite(surv).all()
