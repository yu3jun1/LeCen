"""采样约定：P2/P4/P8 上已知位置的峰值一致；翻转后 to_world 回到同一世界坐标；32³ crop 与直接切片一致；local_grid 每轴范围与 mm 设定一致。"""
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from common.coords import affine_of, patch_to_world, vol_to_patch, vox_to_world
from common.sampling import box_grid, from_level, local_grid, sample, to_level

PATCH = torch.tensor([80.0, 192.0, 192.0])


def test_level_cell_centers():
    g = torch.Generator().manual_seed(0)
    for s in (2, 4, 8):
        shape = [int(n) // s for n in PATCH]
        cell = torch.tensor([torch.randint(n, (1,), generator=g).item() for n in shape])
        feat = torch.zeros(1, 1, *shape)
        feat[(0, 0, *cell)] = 1.0
        u = from_level(cell.float(), s)
        assert torch.allclose(to_level(u, s), cell.float())
        neighbors = torch.stack([u, u + torch.tensor([s, 0.0, 0]), u + torch.tensor([0.0, s, 0]), u + torch.tensor([0.0, 0, s])])
        v = sample(feat, neighbors[None], PATCH)[0, 0]
        assert torch.allclose(v, torch.tensor([1.0, 0, 0, 0]), atol=1e-6)


def test_peak_consistent_across_levels():
    u = torch.tensor([37.5, 101.5, 66.5])
    zz, yy, xx = torch.meshgrid(*[torch.arange(int(n)).float() for n in PATCH], indexing="ij")
    blob = torch.exp(-((zz - u[0]) ** 2 + (yy - u[1]) ** 2 + (xx - u[2]) ** 2) / (2 * 6.0 ** 2))[None, None]
    offs = torch.linspace(-2, 2, 17)
    for s in (2, 4, 8):
        feat = F.avg_pool3d(blob, s)
        for ax in range(3):
            pts = u.repeat(len(offs), 1)
            pts[:, ax] += offs
            peak = offs[sample(feat, pts[None], PATCH)[0, 0].argmax()]
            assert abs(peak) <= 0.5 * s


def test_flip_to_world():
    img = sitk.Image(10, 10, 10, sitk.sitkInt16)
    img.SetSpacing((0.7, 0.7, 1.25))
    img.SetOrigin((-150.0, 20.0, -300.0))
    a = torch.tensor(affine_of(img))
    v = torch.tensor([[40.0, 50.0, 60.0]], dtype=torch.float64)
    off = torch.tensor([5.0, 7.0, 9.0], dtype=torch.float64)
    world = vox_to_world(v, a)
    for f in ([0, 0, 0], [1, 1, 0], [0, 1, 1]):
        flip = torch.tensor(f, dtype=torch.float64)
        assert torch.allclose(patch_to_world(vol_to_patch(v, off, flip, PATCH), off, flip, PATCH, a), world)
    img = torch.rand(1, 1, 80, 192, 192)
    pts = torch.tensor([[12.3, 45.6, 78.9]])
    flipped = img.flip(2, 4)
    pts_f = pts.clone()
    pts_f[:, [0, 2]] = PATCH[[0, 2]] - 1 - pts[:, [0, 2]]
    assert torch.allclose(sample(img, pts[None], PATCH), sample(flipped, pts_f[None], PATCH), atol=1e-6)


def test_box_crop_equals_slicing():
    img = torch.rand(1, 1, 80, 192, 192)
    lo = torch.tensor([20, 100, 33])
    pts = box_grid(lo + 15.5, torch.full((3,), 32.0), 32)
    crop = sample(img, pts[None], PATCH)[0, 0].reshape(32, 32, 32)
    assert torch.allclose(crop, img[0, 0, 20:52, 100:132, 33:65], atol=1e-5)


def test_local_grid_extent_mm():
    sp = torch.tensor([1.25, 0.703125, 0.703125])
    small = local_grid(torch.tensor([40.0, 96, 96]), torch.tensor([2.0, 3, 3]), sp, 7, 1.5, 10.0)
    assert torch.allclose((small.max(0).values - small.min(0).values) * sp, torch.full((3,), 10.0))
    size = torch.tensor([12.0, 30, 25])
    big = local_grid(torch.tensor([40.0, 96, 96]), size, sp, 7, 1.5, 10.0)
    assert torch.allclose((big.max(0).values - big.min(0).values) * sp, 1.5 * size * sp)
