"""在完整体积上提取并缓存 CT-CLIP token。

预处理按 CT-CLIP 官方推理脚本：数组排为 (z, x, y)，trilinear 重采样到 (1.5, 0.75, 0.75) mm，
HU 截断到 [−1000, 1000] 后除以 1000，居中裁剪/以 −1 填充到 (240, 480, 480)。
image encoder eval() + inference_mode()，取池化前 24³ × 512 token，平均池化到 12³，
并按同一组几何变换计算每个池化后 token 中心的世界坐标。
"""
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from common.coords import affine_of, vox_to_world
from common.io import write_npz

SPACING_ZXY = np.array([1.5, 0.75, 0.75])
SHAPE_ZXY = np.array([240, 480, 480])
TOKEN_PATCH_ZXY = np.array([10, 20, 20])
GRID = 24


def load_encoder(ckpt):
    from transformer_maskgit import CTViT

    enc = CTViT(dim=512, codebook_size=8192, image_size=480, patch_size=20, temporal_patch_size=10,
                spatial_depth=4, temporal_depth=4, dim_head=32, heads=8)
    sd = torch.load(ckpt, map_location="cpu")
    p = "visual_transformer."
    enc.load_state_dict({k[len(p):]: v for k, v in sd.items() if k.startswith(p)})
    return enc.eval().cuda()


def preprocess(image):
    """返回 (1, 1, 240, 480, 480) 输入，以及每轴 final → 原始 (z, x, y) 连续索引的映射参数。"""
    vol = sitk.GetArrayFromImage(image).transpose(0, 2, 1).astype(np.float32)  # (z, x, y)
    n = np.array(vol.shape)
    sp = np.array(image.GetSpacing())[[2, 0, 1]]
    n_rs = (n * sp / SPACING_ZXY).astype(int)
    x = F.interpolate(torch.from_numpy(vol)[None, None], size=tuple(n_rs), mode="trilinear", align_corners=False)
    x = x.clamp(-1000, 1000) / 1000
    start = np.maximum((n_rs - SHAPE_ZXY) // 2, 0)
    end = np.minimum(start + SHAPE_ZXY, n_rs)
    x = x[..., start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    before = (SHAPE_ZXY - (end - start)) // 2
    after = SHAPE_ZXY - (end - start) - before
    x = F.pad(x, (before[2], after[2], before[1], after[1], before[0], after[0]), value=-1)
    return x, dict(n=n, n_rs=n_rs, start=start, before=before)


def token_world(image, g, pooled_grid):
    """池化后 token 中心（final 索引）→ 重采样索引 → 原始 (z, x, y) 索引 → 世界坐标；顺序与 token 展平一致。"""
    patch = TOKEN_PATCH_ZXY * (GRID // np.array(pooled_grid))
    idx = np.stack(np.meshgrid(*[np.arange(k) for k in pooled_grid], indexing="ij"), -1).reshape(-1, 3)
    final = idx * patch + (patch - 1) / 2
    rs = final - g["before"] + g["start"]
    orig_zxy = (rs + 0.5) * g["n"] / g["n_rs"] - 0.5
    return vox_to_world(orig_zxy[:, [0, 2, 1]], affine_of(image)).astype(np.float32)


@torch.inference_mode()
def extract(image, encoder, token_layer, pooled_grid):
    x, g = preprocess(image)
    feats = {}
    hook = dict(encoder.named_modules())[token_layer].register_forward_hook(lambda m, i, o: feats.update(t=o))
    encoder(x.cuda(), return_encoded_tokens=True)
    hook.remove()
    t = feats["t"].permute(0, 4, 1, 2, 3).float()  # 要求 (1, 24, 24, 24, 512)，不 reshape 以免打乱 token 顺序
    t = F.adaptive_avg_pool3d(t, tuple(pooled_grid))[0].permute(1, 2, 3, 0).reshape(-1, 512)
    return t.cpu().numpy(), token_world(image, g, pooled_grid)


def cache(path, image, encoder, cfg):
    tokens, xyz = extract(image, encoder, cfg.ctclip.token_layer, cfg.ctclip.pooled_grid)
    write_npz(path, tokens=tokens.astype(cfg.ctclip.dtype), token_world_xyz=xyz, ctclip_ckpt=np.array(str(cfg.ctclip.checkpoint)),
              token_layer=np.array(cfg.ctclip.token_layer), preprocessing_version=np.array(cfg.preprocess.version))
