"""Regression for the saved D1-15441 / D2-10457 FP16 convolution overflows."""
import pytest
import torch

from models.heatmap_head import HeatmapHead


def test_heatmap_fp32_output_and_gradients_are_unchanged():
    torch.manual_seed(19)
    head = HeatmapHead(16)
    p = torch.randn(2, 16, 4, 4, 4, requires_grad=True)
    actual = head(p)
    features = head.conv(p)
    expected = head.hm(features), head.size(features)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(sum(x.square().sum() for x in actual), p)[0]
    expected_grad = torch.autograd.grad(sum(x.square().sum() for x in expected), p)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FP16 CUDA regression")
def test_heatmap_fp16_convolution_overflow_has_finite_forward_and_backward():
    torch.manual_seed(19)
    head = HeatmapHead(16).cuda().train()
    with torch.no_grad():
        head.conv[0].weight.fill_(0.5)
        head.conv[0].bias.zero_()
    p = torch.full((2, 16, 3, 3, 3), 500., device="cuda", requires_grad=True)
    # Finite input and weights produce 108000 at the centre of the convolution.
    # Normalization afterwards cannot recover an already-overflowed FP16 value.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        old_conv = head.conv[0](p)
        assert torch.isinf(old_conv).any()
        assert not torch.isfinite(head.hm(head.conv(p))).all()
    with torch.no_grad():
        expected = head(p)
    with torch.autocast("cuda", dtype=torch.float16):
        hm, size = head(p)
        assert hm.dtype == size.dtype == torch.float16  # AMP remains on for the output heads.
        assert torch.isfinite(hm).all() and torch.isfinite(size).all()
        for actual, reference in zip((hm, size), expected):
            torch.testing.assert_close(actual.float(), reference, rtol=5e-3, atol=5e-3)
        loss = hm.float().square().mean() + size.float().square().mean()
    loss.backward()
    assert torch.isfinite(p.grad).all()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in head.parameters())
    assert head.conv[0].weight.grad.abs().sum() > 0
