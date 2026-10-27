"""体素坐标 → 各层坐标 → grid_sample 归一化坐标。所有 grid_sample 坐标只由本模块生成。

- 体素索引 u（从 0 开始）的中心位于连续坐标 u；patch 尺寸必须能被各层 stride 整除。
- patch 体素坐标 → 第 l 层坐标（每轴）：u_l = (u + 0.5) / s_l − 0.5。
- 归一化坐标（固定 align_corners=False，每轴）：n = 2(u + 0.5) / N_patch − 1；
  由于各层尺寸 × stride = patch 尺寸，同一 patch 体素坐标在 P2/P4/P8 上的归一化坐标相同。
- 张量为 (N, C, D, H, W)，grid 最后一维为 (x, y, z)，由内部 (z, y, x) 转换时显式反转。
- mm → 体素（每轴）：v[a] = mm[a] / sp[a]。
"""
import torch
import torch.nn.functional as F


def to_level(u, stride):
    return (u + 0.5) / stride - 0.5


def from_level(u_l, stride):
    return (u_l + 0.5) * stride - 0.5


def to_grid(c_zyx_vox, patch_zyx):
    n = 2.0 * (c_zyx_vox + 0.5) / patch_zyx - 1.0
    return n.flip(-1)


def mesh3(offs):
    """(k,) → (k³, 3)，zyx。"""
    return torch.stack(torch.meshgrid(offs, offs, offs, indexing="ij"), -1).reshape(-1, 3)


def local_grid(center_zyx, size_zyx, sp, k=7, scale=1.5, min_mm=10.0):
    """(..., 3) 中心与尺寸（patch 体素）→ (..., k³, 3) patch 体素坐标；每轴范围 max(scale·size, min_mm)。"""
    extent_mm = torch.maximum(scale * size_zyx * sp, torch.full_like(size_zyx, min_mm))
    extent_vox = extent_mm / sp
    offs = torch.linspace(-0.5, 0.5, k, device=center_zyx.device)
    return center_zyx[..., None, :] + mesh3(offs) * extent_vox[..., None, :]


def box_grid(center_zyx, size_zyx, out):
    """把框等分为 out³ 个体素，返回各体素中心的 patch 体素坐标 (..., out³, 3)。"""
    offs = (torch.arange(out, device=center_zyx.device) + 0.5) / out - 0.5
    return center_zyx[..., None, :] + mesh3(offs) * size_zyx[..., None, :]


def inside_patch(pts, patch_zyx):
    return ((pts >= -0.5) & (pts <= patch_zyx - 0.5)).all(-1)


def sample(feat, pts, patch_zyx, mode="bilinear"):
    """feat (B, C, D, H, W) 上按 patch 体素坐标 pts (B, *S, 3) 采样 → (B, C, *S)；坐标与采样均为 float32。"""
    b, s = pts.shape[0], pts.shape[1:-1]
    grid = to_grid(pts.float(), patch_zyx).reshape(b, -1, 1, 1, 3)
    out = F.grid_sample(feat.float(), grid, mode=mode, align_corners=False)
    return out.reshape(b, feat.shape[1], *s)
