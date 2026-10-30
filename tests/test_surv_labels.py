"""区间切分、删失编码与 pycox 期望一致；1 − Ŝ(t*) 计算正确；结局编码与合格队列正确。"""
import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from data.make_splits import clin_splits
from data.prep_acrin import harmonize
from data.prep_clinical import clinical_cohort
from data.prep_lung1 import TREATMENT
from losses.surv_loss import fit_label_transform, interp_survival, survival


def test_label_transform():
    rng = np.random.default_rng(0)
    time = rng.uniform(10, 1500, 200).astype(np.float32)
    event = (rng.random(200) < 0.6).astype(np.float32)
    lt = fit_label_transform(10, time, event)
    cuts = lt.cuts
    assert len(cuts) <= 10 and cuts[0] == 0 and np.isclose(cuts[-1], time.max())
    idx, ev = lt.transform(time, event)
    for t, e, i in zip(time, event, idx):
        if e:   # 事件移到右侧切点
            assert cuts[i] >= t and (i == 0 or cuts[i - 1] < t)
        else:   # 删失移到左侧切点
            assert cuts[i] <= t and (i == len(cuts) - 1 or cuts[i + 1] > t)
    assert (ev == event).all()


def test_risk_at_t_star():
    cuts = np.array([0.0, 100, 300, 730, 1000])
    logits = torch.full((2, 5), float(np.log(0.1 / 0.9)))
    s = survival(logits).numpy()
    assert np.allclose(s[0], 0.9 ** np.arange(1, 6))
    t_star = 730.0
    risk = 1 - interp_survival(s, cuts, np.array([t_star]))[:, 0]
    assert np.allclose(risk, 1 - 0.9 ** 4)
    mid = interp_survival(s, cuts, np.array([200.0]))[0, 0]
    assert np.isclose(mid, (0.9 ** 2 + 0.9 ** 3) / 2)


def write_cohort(root, name, clinical, series, tumors=None):
    d = root / name
    d.mkdir()
    clinical.to_csv(d / "clinical.csv", index=False)
    series.to_csv(d / "series.csv", index=False)
    (tumors if tumors is not None else pd.DataFrame(columns=["patient_id", "tumor_id", "bbox_world", "source"])).to_csv(d / "tumors.csv", index=False)
    return {"prep": str(d)}


def cohort_cfg(cohorts):
    return OmegaConf.create({"clinical": {"development": "dev", "covariates": ["age", "stage"], "cohorts": cohorts},
                             "splits": {"clin": {"ratios": [0.6, 0.2, 0.2], "stratify": ["event", "treatment"], "seed": 2026}}})


def dev_table():
    return pd.DataFrame({"patient_id": ["01", "02", "03", "04", "05", "06"], "time_days": [100, 200, 300, None, 500, 600],
                         "event": [1, 0, None, 1, 2, 0], "treatment": ["RT", "RT", "RT", "RT", "RT", None],
                         "age": [60, None, 70, 65, 55, 50], "stage": ["I", "IIIA", "II", "I", "IIIB", None]})


def test_clinical_cohort(tmp_path):
    series = pd.DataFrame({"patient_id": ["01", "02", "03", "04", "05", "06"], "series_uid": list("abcdef")})
    cfg = cohort_cfg({"dev": write_cohort(tmp_path, "dev", dev_table(), series.iloc[[0, 2, 3, 4, 5]])})
    df, excluded = clinical_cohort(cfg)
    assert df.patient_id.tolist() == ["01"] and df.event.tolist() == [1] and df.series_uid.tolist() == ["a"]
    assert dict(zip(excluded.patient_id, excluded.reason)) == {
        "02": "no_baseline_ct", "03": "unknown_outcome", "04": "invalid_time", "05": "unknown_outcome", "06": "missing_treatment"}
    with pytest.raises(ValueError):
        clinical_cohort(cohort_cfg({"dev": write_cohort(tmp_path, "dup", dev_table(), pd.concat([series, series.iloc[:1]]))}))


def test_cohorts_share_covariate_columns(tmp_path):
    """类别协变量在合并后展开：只在外部队列出现的类别也有同一列；跨队列重复编号被拒绝。"""
    dev = dev_table().iloc[:2].assign(event=[1, 0], stage=["I", "IIIA"])
    ext = pd.DataFrame({"patient_id": ["E1", "E2"], "time_days": [50, 80], "event": [1, 1], "treatment": ["RT", "RT"],
                        "age": [70, 71], "stage": ["other", None]})
    cohorts = {"dev": write_cohort(tmp_path, "dev", dev, pd.DataFrame({"patient_id": ["01", "02"], "series_uid": ["a", "b"]})),
               "ext": write_cohort(tmp_path, "ext", ext, pd.DataFrame({"patient_id": ["E1", "E2"], "series_uid": ["c", "d"]}))}
    df, excluded = clinical_cohort(cohort_cfg(cohorts))
    assert excluded.empty and df.cohort.tolist() == ["dev", "dev", "ext", "ext"]
    assert {"stage_I", "stage_IIIA", "stage_other", "stage_missing"} <= set(df.columns)
    assert df.stage_other.tolist() == [0, 0, 1, 0] and df.stage_missing.tolist() == [0, 0, 0, 1]
    ext.patient_id = ["01", "E2"]
    write_cohort(tmp_path, "clash", ext, pd.DataFrame({"patient_id": ["01", "E2"], "series_uid": ["c", "d"]}))
    with pytest.raises(ValueError):
        clinical_cohort(cohort_cfg({"dev": cohorts["dev"], "ext": {"prep": str(tmp_path / "clash")}}))


def test_external_cohort_not_split():
    rng = np.random.default_rng(0)
    cohort = pd.DataFrame({"patient_id": [f"P{i}" for i in range(50)], "cohort": ["dev"] * 40 + ["ext"] * 10,
                           "event": rng.integers(0, 2, 50), "treatment": "RT"})
    s = clin_splits(cohort_cfg({}), cohort)
    assert (s.split[s.cohort == "ext"] == "external").all()
    assert s.split[s.cohort == "dev"].value_counts().to_dict() == {"train": 24, "val": 8, "test": 8}


def test_acrin_harmonize():
    frozen = pd.DataFrame({"patient_id": list("abcde"), "time_days": [1, 2, 3, 4, 5], "event": [1, 0, 1, 1, 0],
                           "treatment": "CRT", "age": [60, 61, 62, 63, 64], "sex_male": [1, 0, 1, 0, 1], "baseline_PS": 0,
                           "baseline_stage_IIIA": [0, 1, 0, 0, 0], "baseline_stage_IIIB": [0, 0, 1, 0, 0],
                           "baseline_stage_Other": [0, 0, 0, 1, 0]})
    for c in ("baseline_stage_IIIA", "baseline_stage_IIIB", "baseline_stage_Other"):
        frozen[f"{c}_missing"] = [0, 0, 0, 0, 1]
    h = harmonize(frozen)
    assert h.stage.tolist()[:4] == ["II", "IIIA", "IIIB", "other"] and pd.isna(h.stage.iloc[4])
    assert (h.treatment == TREATMENT).all() and "baseline_PS" not in h.columns
