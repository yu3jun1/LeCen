"""f / g 更新分支：配置解析为预期分支，forward 实际执行该分支，S5 vs S2 只差更新位置，S5 vs S4 只差 slot 来源。

首轮 S4/S5 用 cfg.update 判断分支，取到的是 DictConfig 的 dict.update 方法，全部执行了 g；这里的测试直接检查执行路径。
"""
import pytest
import torch
from omegaconf import OmegaConf

from common.io import ROOT, load_config
from models.clinical import ClinicalModel

K, D_H, D_G, N_CLIN = 4, 8, 6, 3


def model_cfg(name):
    return load_config(ROOT / "configs" / "clin" / f"{name}.yaml").model


def batch(n=5):
    torch.manual_seed(0)
    mask = torch.ones(n, K, dtype=torch.bool)
    mask[0, 2:] = False
    return {"h": torch.randn(n, K, D_H), "mask": mask, "empty": torch.zeros(n), "g_global": torch.randn(n, D_G),
            "c": torch.randn(n, N_CLIN), "a": torch.zeros(n, dtype=torch.long)}


def build(m):
    torch.manual_seed(1)
    return ClinicalModel(m, D_G, D_H, N_CLIN, 1, 5).eval()


def test_configs_resolve_to_intended_branches():
    assert model_cfg("S2")["update_at"] == "pooled"
    assert model_cfg("S4")["update_at"] == "lesion" and model_cfg("S5")["update_at"] == "lesion"


def test_s4_and_s5_share_the_whole_model_config():
    """S5 vs S4 只比较 D1 与 D2 的 slot：模型结构与训练设置完全相同，差别只来自 slots.extractor。"""
    s4, s5 = (OmegaConf.to_container(load_config(ROOT / "configs" / "clin" / f"{n}.yaml"), resolve=True) for n in ("S4", "S5"))
    for c in (s4, s5):
        c.pop("name", None)
        c["slots"].pop("extractor")
    assert s4 == s5


def test_s5_and_s2_differ_only_in_update_position():
    s2, s5 = (OmegaConf.to_container(model_cfg(n)) for n in ("S2", "S5"))
    assert {k for k in s5 if s5[k] != s2.get(k)} == {"update_at"}


@pytest.mark.parametrize("name, branch, update_ndim", [("S5", "lesion", 3), ("S4", "lesion", 3), ("S2", "pooled", 2)])
def test_forward_executes_configured_branch(name, branch, update_ndim):
    model = build(model_cfg(name))
    seen = []
    model.update.register_forward_hook(lambda mod, inp, out: seen.append(inp[0].dim()))
    model(batch())
    assert model.update_at == model.executed_update == branch
    assert seen == [update_ndim]  # 逐病灶：(B, K, ·)；聚合后：(B, ·)


def test_lesion_and_pooled_updates_give_different_outputs():
    """同一组参数（两种结构参数形状相同）下，多 slot 输入的输出必须不同；首轮 S5 与 S2 的预测逐字节相同。"""
    lesion, pooled = build(model_cfg("S5")), build(model_cfg("S2"))
    pooled.load_state_dict(lesion.state_dict())
    b = batch()
    with torch.no_grad():
        assert not torch.allclose(lesion(b), pooled(b))


def test_unknown_update_position_is_rejected():
    m = OmegaConf.merge(model_cfg("S5"), {"update_at": "f"})
    with pytest.raises(ValueError):
        build(m)
