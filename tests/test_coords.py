"""体素 ↔ 世界坐标在裁剪、重采样、翻转下往返误差 < 0.5 体素。"""
import numpy as np
import SimpleITK as sitk
import torch
from omegaconf import OmegaConf

from common.coords import affine_of, flip_vox, patch_to_vol, patch_to_world, vol_to_patch, vox_to_vox, vox_to_world, world_to_vox
from data.prep_luna16 import crop_resample

PP = OmegaConf.create({"spacing_zyx": [1.25, 0.703125, 0.703125], "hu_clip": [-1000, 400]})


def raw_image(blob_zyx):
    arr = np.full((60, 120, 110), -1000, np.int16)
    z, y, x = blob_zyx
    arr[z - 1:z + 2, y - 1:y + 2, x - 1:x + 2] = 400
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.8, 0.65, 2.5))
    img.SetOrigin((-120.0, 35.5, -310.0))
    c, s = np.cos(0.3), np.sin(0.3)
    img.SetDirection((c, -s, 0, s, c, 0, 0, 0, 1))
    return img


def centroid(arr):
    w = arr.astype(float) + 1000
    return (np.argwhere(np.ones_like(arr, bool)) * w.reshape(-1, 1)).sum(0) / w.sum()


def test_vox_world_roundtrip():
    a = affine_of(raw_image((30, 60, 55)))
    v = np.random.default_rng(0).uniform(0, 100, (50, 3))
    assert np.abs(world_to_vox(vox_to_world(v, a), a) - v).max() < 1e-9
    img = raw_image((30, 60, 55))
    for v_ in v[:5]:
        w = img.TransformContinuousIndexToPhysicalPoint(v_[::-1].tolist())
        assert np.allclose(vox_to_world(v_, a)[0], w)


def test_crop_resample_roundtrip():
    blob = (30, 60, 55)
    img = raw_image(blob)
    out = crop_resample(img, np.array([10, 20, 15]), np.array([55, 110, 100]), PP)
    pred = vox_to_vox(np.array(blob, float), affine_of(img), affine_of(out))[0]
    assert np.abs(pred - centroid(sitk.GetArrayFromImage(out))).max() < 0.5
    back = vox_to_vox(pred, affine_of(out), affine_of(img))[0]
    assert np.abs(back - blob).max() < 1e-6


def test_flip_roundtrip():
    patch = torch.tensor([80.0, 192.0, 192.0])
    a = torch.tensor(affine_of(raw_image((30, 60, 55))), dtype=torch.float64)
    v = torch.tensor([[33.2, 101.7, 17.0]], dtype=torch.float64)
    off = torch.tensor([10.0, 20.0, -30.0], dtype=torch.float64)
    for f in ([0, 0, 0], [1, 0, 1], [1, 1, 1]):
        flip = torch.tensor(f, dtype=torch.float64)
        u = vol_to_patch(v, off, flip, patch)
        assert torch.allclose(patch_to_vol(u, off, flip, patch), v)
        assert torch.allclose(patch_to_world(u, off, flip, patch, a), vox_to_world(v, a))
        assert torch.allclose(flip_vox(flip_vox(u, flip, patch), flip, patch), u)
