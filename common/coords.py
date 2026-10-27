"""体素 ↔ 世界坐标；裁剪/重采样/翻转的正逆变换。

体素坐标一律 (z, y, x)，世界坐标一律 SimpleITK 物理坐标 (x, y, z)，单位 mm。
仿射 A (4×4) 把 [z, y, x, 1] 映射到 [x, y, z, 1]，含 direction，所以非单位 direction 在此统一处理。
裁剪与重采样都由 SimpleITK 写入图像头，不同网格之间经世界坐标换算（vox_to_vox）。
函数只用算术运算，numpy 数组与 torch 张量通用；点的形状为 (..., N, 3)，仿射为 (..., 4, 4)。
"""
import numpy as np
import torch


def affine_of(img):
    d = np.array(img.GetDirection()).reshape(3, 3)
    a = np.eye(4)
    a[:3, :3] = (d * np.array(img.GetSpacing()))[:, ::-1]
    a[:3, 3] = img.GetOrigin()
    return a


def vox_to_world(v, a):
    return v @ a[..., :3, :3].swapaxes(-1, -2) + a[..., None, :3, 3]


def world_to_vox(w, a):
    xp = torch if isinstance(a, torch.Tensor) else np
    return (w - a[..., None, :3, 3]) @ xp.linalg.inv(a[..., :3, :3]).swapaxes(-1, -2)


def vox_to_vox(v, a_src, a_dst):
    return world_to_vox(vox_to_world(v, a_src), a_dst)


def flip_vox(u, flip, patch_zyx):
    """patch 内按轴翻转（flip 为 0/1）；自身即逆变换。"""
    return u + flip * (patch_zyx - 1 - 2 * u)


def patch_to_vol(u, offset_vox, flip, patch_zyx):
    """patch 体素坐标 → 撤销翻转 → 加 offset_vox → 预处理体积体素坐标。"""
    return flip_vox(u, flip, patch_zyx) + offset_vox


def vol_to_patch(v, offset_vox, flip, patch_zyx):
    return flip_vox(v - offset_vox, flip, patch_zyx)


def patch_to_world(u, offset_vox, flip, patch_zyx, a):
    return vox_to_world(patch_to_vol(u, offset_vox, flip, patch_zyx), a)
