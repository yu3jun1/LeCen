"""Query decoder：自注意力 / 局部邻域交叉注意力 / G 交叉注意力 / FFN / 逐层框细化。

初始化：q_content = Linear(sample(P[L], ref_center)) + query_embed；q_pos = PE(to_world(ref_center))。
注意力只取输出（need_weights=False），不在 fp16 下显式写出注意力 logit。
每层：参考框 detach 后用新中心重新计算 q_pos，供下一层的自注意力与 G 注意力使用。
to_world：patch 体素坐标 → 撤销翻转 → 加 offset_vox → 预处理体积的 affine → 世界坐标。
"""
import torch
import torch.nn as nn

from common.coords import patch_to_world
from common.sampling import local_grid, mesh3, sample
from models.det_head import DetHead

LEVELS = (2, 4, 8)


class DecoderLayer(nn.Module):
    def __init__(self, dim, heads, k):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.local_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.global_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(4))
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.level_embed = nn.Parameter(torch.randn(len(LEVELS), 1, dim) * 0.02)
        self.rel_pos = nn.Linear(3, dim)
        self.register_buffer("offs", mesh3(torch.linspace(-0.5, 0.5, k)), persistent=False)
        self.head = DetHead(dim)

    def forward(self, q, q_pos, P, G, pts, patch):
        b, n, d = q.shape
        x = q + q_pos
        q = self.norms[0](q + self.self_attn(x, x, q, need_weights=False)[0])

        feats = torch.cat([sample(P[s], pts, patch).permute(0, 2, 3, 1) + e for s, e in zip(LEVELS, self.level_embed)], 2)
        keys = feats + self.rel_pos(self.offs).repeat(len(LEVELS), 1)
        local = self.local_attn((q + q_pos).reshape(b * n, 1, d), keys.flatten(0, 1), feats.flatten(0, 1), need_weights=False)[0]
        q = self.norms[1](q + local.reshape(b, n, d))

        if G is not None:
            q = self.norms[2](q + self.global_attn(q + q_pos, G, G, need_weights=False)[0])
        return self.norms[3](q + self.ffn(q))


class QueryDecoder(nn.Module):
    def __init__(self, cfg, pe):
        super().__init__()
        m, d = cfg.model, cfg.model.decoder
        self.cfg = d
        self.pe = pe
        self.query_embed = nn.Embedding(m.heatmap.n_q, m.dim)
        self.content = nn.Linear(m.dim, m.dim)
        self.layers = nn.ModuleList(DecoderLayer(m.dim, d.heads, d.local_grid) for _ in range(d.layers))
        self.register_buffer("patch", torch.tensor(list(cfg.patch_zyx), dtype=torch.float32), persistent=False)

    def to_world(self, center, meta):
        return patch_to_world(center, meta["offset_vox"][:, None], meta["flip"][:, None], self.patch, meta["affine"])

    def forward(self, P, G, ref_center, ref_size, meta, level):
        d, sp = self.cfg, meta["spacing"][:, None]
        q = self.content(sample(P[level], ref_center, self.patch).transpose(1, 2)) + self.query_embed.weight
        center, size = ref_center, ref_size
        q_pos = self.pe(self.to_world(center, meta))
        outs = []
        for layer in self.layers:
            pts = local_grid(center, size, sp, d.local_grid, d.local_extent_scale, d.local_extent_min_mm)
            q = layer(q, q_pos, P, G, pts, self.patch)
            exist_logit, box = layer.head(q, center, size, self.patch)
            outs.append({"h": q, "exist_logit": exist_logit, "box": box})
            center, size = box[..., :3].detach(), box[..., 3:].detach()
            q_pos = self.pe(self.to_world(center, meta))
        return outs
