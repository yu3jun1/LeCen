"""区间切分、删失编码与 pycox 期望一致；1 − Ŝ(t*) 计算正确；结局编码与合格队列正确。"""
import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from data.prep_clinical import clinical_cohort
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


def cohort_cfg(tmp_path, table, series):
    table.to_csv(tmp_path / "clin.csv", index=False)
    series.to_csv(tmp_path / "series.csv", index=False)
    return OmegaConf.create({"paths": {"clinical_table": str(tmp_path / "clin.csv"), "clinical_series": str(tmp_path / "series.csv")},
                             "clinical": {"columns": {"patient_id": "pid", "time_days": "t", "event": "status", "treatment": "arm"},
                                          "event_values": ["dead"], "censor_values": ["alive"], "covariates": ["age"]}})


def test_clinical_cohort(tmp_path):
    table = pd.DataFrame({"pid": ["01", "02", "03", "04", "05", "06"], "t": [100, 200, 300, None, 500, 600],
                          "status": ["dead", "alive", "unknown", "dead", None, "alive"],
                          "arm": ["A", "B", "A", "A", "B", None], "age": [60, None, 70, 65, 55, 50]})
    series = pd.DataFrame({"patient_id": ["01", "02", "03", "04", "05", "06"], "series_uid": list("abcdef")})
    df, excluded = clinical_cohort(cohort_cfg(tmp_path, table, series.iloc[[0, 2, 3, 4, 5]]))
    assert df.patient_id.tolist() == ["01"] and df.event.tolist() == [1] and df.series_uid.tolist() == ["a"]
    assert dict(zip(excluded.patient_id, excluded.reason)) == {
        "02": "no_baseline_ct", "03": "unknown_outcome", "04": "invalid_time", "05": "unknown_outcome", "06": "missing_treatment"}
    df, _ = clinical_cohort(cohort_cfg(tmp_path, table, series))
    assert dict(zip(df.patient_id, df.event)) == {"01": 1, "02": 0} and df.age_missing.tolist() == [0, 1]
    with pytest.raises(ValueError):
        clinical_cohort(cohort_cfg(tmp_path, table, pd.concat([series, series.iloc[:1]])))
