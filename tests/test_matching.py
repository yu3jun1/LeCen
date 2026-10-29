"""构造的框能得到预期匹配；无 GT 时返回空匹配；忽略区域规则正确；未受监督位置的极端输出下匹配代价仍有限。"""
import torch
from omegaconf import OmegaConf

from losses.det_loss import layer_loss
from losses.matching import hungarian_match
from common.boxes import size_from_log
from models.det_head import DetHead
from models.heatmap_head import HeatmapHead

CFG = OmegaConf.create({"loss": {"match_cost": {"cls": 2.0, "l1": 5.0, "giou": 2.0},
                                 "exist_focal": {"alpha": 0.25, "gamma": 2.0}}})
PATCH = torch.tensor([80.0, 192.0, 192.0])


def boxes():
    return torch.tensor([[20.0, 50, 60, 8, 10, 10], [60.0, 150, 30, 4, 6, 6], [40.0, 96, 96, 12, 20, 20]])


def test_expected_match():
    gt = boxes()
    perm = torch.tensor([2, 0, 1])
    pred = torch.cat([gt[perm] + 0.3, torch.tensor([[10.0, 10, 10, 3, 3, 3]] * 5)])
    logit = torch.full((8,), -2.0)
    row, col = hungarian_match(logit, pred, gt, PATCH, CFG.loss.match_cost, CFG.loss.exist_focal)
    assert dict(zip(row.tolist(), col.tolist())) == {0: 2, 1: 0, 2: 1}


def test_no_gt():
    row, col = hungarian_match(torch.zeros(4), boxes()[[0, 0, 1, 2]], torch.zeros(0, 6), PATCH,
                               CFG.loss.match_cost, CFG.loss.exist_focal)
    assert len(row) == len(col) == 0


def test_ignore_rule():
    pred = torch.cat([boxes()[:1], torch.tensor([[60.0, 150, 30, 4, 6, 6]])])
    layer = {"exist_logit": torch.tensor([[0.5, 2.0]]), "box": pred[None]}
    gt = boxes()[:1]
    match = [(torch.tensor([0]), torch.tensor([0]))]
    batch = {"boxes": [gt], "ignore": [torch.zeros(0, 6)]}
    base = layer_loss(layer, batch, match, CFG, PATCH, partial=False)[0]
    batch["ignore"] = [torch.tensor([[60.0, 150, 30, 5, 5, 5]])]
    ign = layer_loss(layer, batch, match, CFG, PATCH, partial=False)[0]
    only_pos = layer_loss({"exist_logit": layer["exist_logit"][:, :1], "box": pred[None, :1]}, batch, match, CFG, PATCH, False)[0]
    assert ign < base and torch.isclose(ign, only_pos)
    # 匹配的正例优先：即使中心在忽略区域内也计正例损失
    batch["ignore"] = [torch.tensor([[20.0, 50, 60, 5, 5, 5]])]
    assert torch.isclose(layer_loss(layer, batch, match, CFG, PATCH, False)[0], base)


def test_extreme_outputs_keep_cost_finite():
    """候选点 log 尺寸与逐层 Δlog 尺寸极大时（此前 GIoU 中 inf − inf → NaN，匹配报 invalid numeric entries）。"""
    hm = torch.zeros(1, 1, 20, 48, 48)
    hm[0, 0, 10, 24, 24] = 5.0
    ref_c, ref_s = HeatmapHead.candidates(hm, torch.full((1, 3, 20, 48, 48), 1e4), 4, 8, 3)
    assert torch.equal(ref_s, (2 * PATCH).expand_as(ref_s))
    head = DetHead(16)
    torch.nn.init.constant_(head.box[-1].bias, 200.0)
    exist, box = head(torch.randn(1, 8, 16), ref_c, ref_s, PATCH)
    assert torch.isfinite(box).all() and (box[0, :, 3:] == 2 * PATCH).all()
    row, col = hungarian_match(exist[0], box[0], boxes(), PATCH, CFG.loss.match_cost, CFG.loss.exist_focal)
    assert len(row) == 3
    torch.nn.init.constant_(head.box[-1].bias, -200.0)
    assert (head(torch.randn(1, 8, 16), ref_c, ref_s, PATCH)[1][0, :, 3:] == 1).all()


def test_size_bounds_preserve_in_range_gradients_and_do_not_mask_nan():
    log_size = torch.tensor([[-1000., torch.log(torch.tensor(8.)).item(), 1000.]], requires_grad=True)
    size = size_from_log(log_size, PATCH)
    torch.testing.assert_close(size, torch.tensor([[1., 8., 384.]]))
    size.sum().backward()
    torch.testing.assert_close(log_size.grad, torch.tensor([[0., 8., 0.]]))
    assert torch.isnan(size_from_log(torch.full((1, 3), float("nan")), PATCH)).all()
