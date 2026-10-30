"""四种空样本情况：损失有限、不报错、计数正确。"""
import torch
from omegaconf import OmegaConf

from losses.det_loss import det_losses
from losses.heatmap_loss import heatmap_loss
from losses.recovery_loss import recovery_term
from models.recovery import RecoveryDecoder

PATCH = torch.tensor([80.0, 192.0, 192.0])
CFG = OmegaConf.create({"loss": {"exist_focal": {"alpha": 0.25, "gamma": 2.0}, "w": {"exist": 2.0, "l1": 5.0, "giou": 2.0}},
                        "model": {"recovery": {"out_size": 32, "crop_mm": 32, "dilate_vox": 2}}})


class Counting(RecoveryDecoder):
    calls = 0

    def forward(self, h):
        Counting.calls += 1
        return super().forward(h)


def out_and_batch(gt, has_mask, valid_all=True):
    n = 6
    box = torch.cat([gt, torch.tensor([[10.0, 10, 10, 4, 4, 4]] * (n - len(gt)))])[None].requires_grad_()
    out = {"hm": torch.randn(1, 1, 20, 48, 48), "size": torch.randn(1, 3, 20, 48, 48),
           "layers": [{"h": torch.randn(1, n, 16, requires_grad=True), "exist_logit": torch.randn(1, n, requires_grad=True), "box": box}]}
    valid = torch.ones(1, 80, 192, 192, dtype=torch.bool) if valid_all else torch.zeros(1, 80, 192, 192, dtype=torch.bool)
    labels = torch.zeros(1, 80, 192, 192, dtype=torch.long)
    labels[0, 18:22, 48:52, 58:62] = 1
    batch = {"boxes": [gt], "ignore": [torch.zeros(0, 6)], "has_mask": [torch.tensor(has_mask, dtype=torch.bool)],
             "label_value": [torch.ones(len(gt), dtype=torch.long)], "image": torch.rand(1, 1, 80, 192, 192),
             "labels": labels, "valid_mask": valid, "spacing": torch.tensor([[1.25, 0.703125, 0.703125]])}
    match = [(torch.arange(len(gt)), torch.arange(len(gt)))]
    return out, batch, match


def test_no_gt():
    out, batch, match = out_and_batch(torch.zeros(0, 6), [])
    l_hm, l_size = heatmap_loss(out["hm"], out["size"], batch, 4)
    loss, parts = det_losses(out, batch, [match], CFG, PATCH)
    assert torch.isfinite(l_hm) and l_hm > 0 and float(l_size) == 0
    assert torch.isfinite(parts["exist"]) and float(parts["l1"]) == 0 and float(parts["giou"]) == 0
    loss.backward()


def test_gt_without_mask():
    Counting.calls = 0
    out, batch, match = out_and_batch(torch.tensor([[20.0, 50, 60, 4, 4, 4]]), [False])
    assert recovery_term(Counting(16), out, batch, match, CFG, PATCH) == (0.0, 0, 0)
    assert Counting.calls == 0


def test_mask_but_no_valid_voxels():
    out, batch, match = out_and_batch(torch.tensor([[20.0, 50, 60, 4, 4, 4]]), [True], valid_all=False)
    loss, used, zero = recovery_term(Counting(16), out, batch, match, CFG, PATCH)
    assert (loss, used, zero) == (0.0, 0, 1)
    out, batch, match = out_and_batch(torch.tensor([[20.0, 50, 60, 4, 4, 4]]), [True])
    loss, used, zero = recovery_term(Counting(16), out, batch, match, CFG, PATCH)
    assert torch.isfinite(loss) and (used, zero) == (1, 0)
    loss.backward()
    assert out["layers"][0]["h"].grad.abs().sum() > 0


def test_batch_without_recoverable():
    Counting.calls = 0
    out, batch, _ = out_and_batch(torch.tensor([[20.0, 50, 60, 4, 4, 4]]), [True])
    empty = [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))]
    assert recovery_term(Counting(16), out, batch, empty, CFG, PATCH) == (0.0, 0, 0)
    assert Counting.calls == 0
