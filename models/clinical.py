"""临床模块：f（逐病灶更新）、g（聚合后更新）、门控注意力池化、e(a)、e(c)、SurvivalHead。

更新位置由 model.update_at 决定：lesion = f（S4 / S5），pooled = g（S2）。键名不用 update：OmegaConf 的 DictConfig
是 MutableMapping，cfg.update 取到的是 dict 的 update 方法而不是配置值（首轮 S4/S5 因此全部走了 g）。
构造时解析并校验取值；forward 把实际执行的分支写入 executed_update，训练脚本将其与配置一起记录。

g_global 为所有影像配置共用的 G0 全局向量（CT-CLIP token 均值，d_g = 512），h_i 为病灶 slot（d_h = 256）。
S5 / S4：r = pool([h_i + f(h_i, g_global, e(a), e(c))])
S2：     r0 = pool([h_i])；r = r0 + g(r0, g_global, e(a), e(c))（g 与 f 同构）
所有配置空集合时 r = 0、empty = 1（S2 的 g 不作用于空集合，与 S5 一致）。
S3：     S5 去掉所有治疗输入（f 与 head 都不输入 e(a)）
S1（G0）：logits = head([g_global; c; e(a)])，不使用检测器与 slot
其余：   logits = head([r; m; g_global; c; e(a)])，m 为空集合指示
"""
import torch
import torch.nn as nn


def mlp(din, hidden, dout, dropout):
    return nn.Sequential(nn.Linear(din, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dout))


class GatedAttnPool(nn.Module):
    """门控注意力 MIL；空集合时 r = 0。"""

    def __init__(self, dim, hidden=128):
        super().__init__()
        self.v, self.u, self.w = nn.Linear(dim, hidden), nn.Linear(dim, hidden), nn.Linear(hidden, 1)

    def forward(self, h, mask):
        s = self.w(torch.tanh(self.v(h)) * torch.sigmoid(self.u(h))).squeeze(-1)
        a = torch.softmax(s.masked_fill(~mask, -1e9), 1) * mask
        return (a[..., None] * h).sum(1)


class ClinicalModel(nn.Module):
    def __init__(self, m, global_dim, slot_dim, n_clin, n_treat, n_t):
        """slot_dim 只在 m.slots 时使用。"""
        super().__init__()
        self.slots = bool(m.slots)
        self.update_at = m["update_at"] if self.slots else None
        if self.slots and self.update_at not in ("lesion", "pooled"):
            raise ValueError(f"model.update_at 必须为 lesion 或 pooled，得到 {self.update_at!r}")
        self.executed_update = None
        ea = m.ea_dim if m.treatment else 0
        self.ea = nn.Embedding(n_treat, m.ea_dim) if m.treatment else None
        head_in = global_dim + n_clin + ea
        if m.slots:
            self.ec = nn.Linear(n_clin, m.ec_dim)
            self.update = mlp(slot_dim + global_dim + ea + m.ec_dim, m.f_hidden, slot_dim, m.dropout)  # f 或 g
            self.pool = GatedAttnPool(slot_dim)
            head_in += slot_dim + 1
        self.head = mlp(head_in, m.head_hidden, n_t, m.dropout)

    def forward(self, b):
        ea = [self.ea(b["a"])] if self.ea is not None else []
        parts = [b["g_global"], b["c"], *ea]
        if self.slots:
            cond = torch.cat([b["g_global"], *ea, self.ec(b["c"])], -1)
            h, mask = b["h"], b["mask"]
            if self.update_at == "lesion":
                r = self.pool(h + self.update(torch.cat([h, cond[:, None].expand(-1, h.shape[1], -1)], -1)), mask)
                self.executed_update = "lesion"
            else:
                r0 = self.pool(h, mask)
                r = (r0 + self.update(torch.cat([r0, cond], -1))) * mask.any(1, keepdim=True)
                self.executed_update = "pooled"
            parts = [r, b["empty"][:, None], *parts]
        return self.head(torch.cat(parts, -1))
