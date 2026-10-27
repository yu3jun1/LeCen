"""32³ crop 与原始 mask 对齐；有效体素同时排除 patch 外与越界填充。"""
import torch

from losses.recovery_loss import recovery_targets

PATCH = torch.tensor([80.0, 192.0, 192.0])


def test_aligned_crop():
    image = torch.rand(1, 80, 192, 192)
    labels = torch.zeros(80, 192, 192, dtype=torch.long)
    labels[20:52, 100:132, 33:65] = 7
    valid = torch.ones(80, 192, 192, dtype=torch.bool)
    x, msk, val = recovery_targets(image, labels, valid, torch.tensor([35.5, 115.5, 48.5]), torch.tensor([32.0, 32, 32]),
                                   7, PATCH, 32, 1.0)
    assert torch.allclose(x, image[0, 20:52, 100:132, 33:65], atol=1e-5)
    assert (msk == 1).all() and (val == 1).all()


def test_valid_excludes_outside_and_padding():
    image = torch.rand(1, 80, 192, 192)
    labels = torch.zeros(80, 192, 192, dtype=torch.long)
    labels[0:8, 10:26, 10:26] = 3
    valid = torch.ones(80, 192, 192, dtype=torch.bool)
    valid[:, :, :12] = False  # x < 12 为越界填充
    center, size = torch.tensor([3.5, 17.5, 17.5]), torch.tensor([8.0, 16, 16])
    x, msk, val = recovery_targets(image, labels, valid, center, size, 3, PATCH, 32, 1.5)
    pts_z = center[0] + ((torch.arange(32) + 0.5) / 32 - 0.5) * size[0] * 1.5
    pts_x = center[2] + ((torch.arange(32) + 0.5) / 32 - 0.5) * size[2] * 1.5
    z_out = pts_z < -0.5
    x_pad = pts_x.round() < 12
    assert (val[z_out] == 0).all() and (val[:, :, x_pad] == 0).all()
    assert (val[~z_out][:, :, ~x_pad] == 1).all()
    assert msk[~z_out][:, :, ~x_pad].sum() > 0
