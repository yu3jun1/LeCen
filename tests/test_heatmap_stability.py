"""BF16 heatmap convolution must handle activations beyond the FP16 range."""
import pytest
import torch
from models.heatmap_head import HeatmapHead


@pytest.mark.skipif(not torch.cuda.is_available(), reason='BF16 CUDA regression')
def test_heatmap_bfloat16_large_convolution_has_finite_forward_and_backward():
    torch.manual_seed(19)
    head = HeatmapHead(16).cuda().train()
    with torch.no_grad():
        head.conv[0].weight.fill_(0.5)
        head.conv[0].bias.zero_()
    p = torch.full((2, 16, 3, 3, 3), 500., device='cuda', requires_grad=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        conv = head.conv[0](p)
        assert conv.dtype == torch.bfloat16 and conv.max() > 65504
        assert conv.isfinite().all()
        hm, size = head(p)
        assert hm.dtype == size.dtype == torch.bfloat16
        assert hm.isfinite().all() and size.isfinite().all()
        loss = hm.float().square().mean() + size.float().square().mean()
    loss.backward()
    assert torch.isfinite(p.grad).all()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in head.parameters())
    assert head.conv[0].weight.grad.abs().sum() > 0
