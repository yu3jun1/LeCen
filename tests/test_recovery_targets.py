"""32³ crop 与原始 mask 对齐；有效体素同时排除 patch 外与越界填充；目标为固定物理尺寸、保留病灶大小。"""
import torch
from omegaconf import OmegaConf

from eval.shuffle_h import lesion_metrics, template_prediction
from losses.recovery_loss import recovery_inputs, recovery_targets

PATCH = torch.tensor([80.0, 192.0, 192.0])
SPACING = torch.tensor([1.25, 0.703125, 0.703125])


def test_aligned_crop():
    image = torch.rand(1, 80, 192, 192)
    labels = torch.zeros(80, 192, 192, dtype=torch.long)
    labels[20:52, 100:132, 33:65] = 7
    valid = torch.ones(80, 192, 192, dtype=torch.bool)
    x, msk, val = recovery_targets(image, labels, valid, torch.tensor([35.5, 115.5, 48.5]), torch.tensor([32.0, 32, 32]),
                                   7, PATCH, 32)
    assert torch.allclose(x, image[0, 20:52, 100:132, 33:65], atol=1e-5)
    assert (msk == 1).all() and (val == 1).all()


def test_valid_excludes_outside_and_padding():
    image = torch.rand(1, 80, 192, 192)
    labels = torch.zeros(80, 192, 192, dtype=torch.long)
    labels[0:8, 10:26, 10:26] = 3
    valid = torch.ones(80, 192, 192, dtype=torch.bool)
    valid[:, :, :12] = False  # x < 12 为越界填充
    center, crop = torch.tensor([3.5, 17.5, 17.5]), torch.tensor([12.0, 24, 24])
    x, msk, val = recovery_targets(image, labels, valid, center, crop, 3, PATCH, 32)
    pts_z = center[0] + ((torch.arange(32) + 0.5) / 32 - 0.5) * crop[0]
    pts_x = center[2] + ((torch.arange(32) + 0.5) / 32 - 0.5) * crop[2]
    z_out = pts_z < -0.5
    x_pad = pts_x.round() < 12
    assert (val[z_out] == 0).all() and (val[:, :, x_pad] == 0).all()
    assert (val[~z_out][:, :, ~x_pad] == 1).all()
    assert msk[~z_out][:, :, ~x_pad].sum() > 0


def test_fixed_physical_crop_keeps_lesion_size():
    """同一 crop_mm 下，小病灶在目标中占比小、大病灶占比大；按框缩放时两者会相同。"""
    cfg = OmegaConf.create({"model": {"recovery": {"crop_mm": 32, "out_size": 32}}})
    labels = torch.zeros(80, 192, 192, dtype=torch.long)
    labels[38:42, 44:52, 44:52] = 1        # 约 5 × 5.6 × 5.6 mm
    labels[30:50, 112:148, 112:148] = 2    # 约 25 × 25 × 25 mm
    boxes = torch.tensor([[39.5, 47.5, 47.5, 4, 8, 8], [39.5, 129.5, 129.5, 20, 36, 36]])
    batch = {"boxes": [boxes], "has_mask": [torch.tensor([True, True])], "label_value": [torch.tensor([1, 2])],
             "image": torch.rand(1, 1, 80, 192, 192), "labels": labels[None], "valid_mask": torch.ones(1, 80, 192, 192, dtype=torch.bool),
             "spacing": SPACING[None]}
    out = {"layers": [{"h": torch.randn(1, 4, 16)}]}
    _, targets = recovery_inputs(out, batch, [(torch.tensor([0, 1]), torch.tensor([0, 1]))], cfg, PATCH)
    small, large = (t[1].mean().item() for t in targets)
    assert 0 < small < 0.01 < 0.3 < large < 0.6


def test_template_matches_identical_targets():
    msk = torch.zeros(32, 32, 32)
    msk[8:24, 8:24, 8:24] = 1
    t = (torch.rand(32, 32, 32), msk, torch.ones(32, 32, 32))
    pred = template_prediction([t, t])
    dice, l1 = lesion_metrics(pred, *t, 2)
    assert dice == 1.0 and l1 < 1e-4
