"""框格式转换、3D IoU / GIoU。

模型内部框：中心 (cz, cy, cx) + 尺寸 (dz, dy, dx)，单位为当前 patch 的体素。
世界框 box_world 与 MONAI 的 cccwhd 一致：中心 (x, y, z) + 尺寸 (w, h, d)，单位 mm。
"""
import torch

from common.coords import patch_to_world

ZYX_TO_XYZ = [2, 1, 0]


def corners(b):
    return b[..., :3] - b[..., 3:] / 2, b[..., :3] + b[..., 3:] / 2


def _vol(lo, hi):
    return (hi - lo).clamp(min=0).prod(-1)


def iou_giou(a, b):
    """可广播的逐元素 IoU 与 GIoU；成对矩阵用 a[:, None]、b[None]。"""
    alo, ahi = corners(a)
    blo, bhi = corners(b)
    inter = _vol(torch.maximum(alo, blo), torch.minimum(ahi, bhi))
    union = a[..., 3:].prod(-1) + b[..., 3:].prod(-1) - inter
    hull = _vol(torch.minimum(alo, blo), torch.maximum(ahi, bhi))
    iou = inter / union
    return iou, iou - (hull - union) / hull


def iou3d(a, b):
    return iou_giou(a, b)[0]


def giou3d(a, b):
    return iou_giou(a, b)[1]


def norm_box(b, patch_zyx):
    """L1 使用的归一化：中心 / patch 尺寸，尺寸取 log。"""
    return torch.cat([b[..., :3] / patch_zyx, b[..., 3:].log()], -1)


def patch_box_to_world(b, offset_vox, flip, patch_zyx, affine, sp_zyx):
    center = patch_to_world(b[..., :3], offset_vox, flip, patch_zyx, affine)
    return torch.cat([center, (b[..., 3:] * sp_zyx)[..., ZYX_TO_XYZ]], -1)


def luna_to_monai(center_xyz, diameter_mm):
    """LUNA16 参考结节（世界中心 + 直径）→ MONAI cccwhd 世界框。"""
    return [*map(float, center_xyz), *[float(diameter_mm)] * 3]
