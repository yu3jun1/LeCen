"""BF16 attention handles the high-score case from earlier numerical failures."""
import pytest
import torch

from models.query_decoder import DecoderLayer


def force_weights(module, args, kwargs):
    return args, {**kwargs, "need_weights": True}


def inputs(device, value=None):
    def tensor(shape):
        return torch.randn(shape, device=device) if value is None else torch.full(shape, value, device=device)
    return (tensor((1, 2, 16)), tensor((1, 2, 16)),
            {s: tensor((1, 16, 2, 2, 2)) for s in (2, 4, 8)}, tensor((1, 4, 16)),
            torch.full((1, 2, 1, 3), 1.5, device=device), torch.full((3,), 4., device=device))


def test_attention_outputs_agree_for_regular_fp32_inputs():
    torch.manual_seed(13)
    layer = DecoderLayer(16, 4, 1).eval()
    args = inputs("cpu")
    with torch.no_grad():
        actual = layer(*args)
        hooks = [m.register_forward_pre_hook(force_weights, with_kwargs=True)
                 for m in (layer.self_attn, layer.local_attn, layer.global_attn)]
        expected = layer(*args)
    for h in hooks:
        h.remove()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BF16 CUDA regression")
def test_decoder_bfloat16_high_scores_have_finite_forward_and_backward():
    layer = DecoderLayer(16, 4, 1).cuda().train()
    with torch.no_grad():
        for m in (layer.self_attn, layer.local_attn, layer.global_attn):
            m.in_proj_weight.copy_(torch.eye(16, device="cuda").repeat(3, 1))
            m.in_proj_bias.zero_()
            m.out_proj.weight.copy_(torch.eye(16, device="cuda"))
            m.out_proj.bias.zero_()
        layer.level_embed.zero_()
        layer.rel_pos.weight.zero_()
        layer.rel_pos.bias.zero_()
    args = inputs("cuda", 300.)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = layer(*args)
        assert torch.isfinite(actual).all()
        loss = actual.float().square().mean()
    loss.backward()
    gradients = [p.grad for p in layer.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
