"""区间切分、删失编码与 pycox 期望一致；1 − Ŝ(t*) 计算正确。"""
import numpy as np
import torch

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
