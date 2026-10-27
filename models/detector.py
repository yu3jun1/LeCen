"""组装 Detector(cfg)。恢复 decoder 不在 forward 中，由训练循环在匹配后调用，D2 不构建该模块。"""
import torch.nn as nn

from models.backbone import ResNet18FPN
from models.global_adapter import GlobalAdapter, PosEnc
from models.heatmap_head import HeatmapHead
from models.query_decoder import QueryDecoder


class Detector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        self.cfg = cfg
        self.pe = PosEnc(m.dim)
        self.backbone = ResNet18FPN(m.dim)
        self.adapter = GlobalAdapter(m.dim, self.pe) if m.decoder.use_global else None
        self.heatmap = HeatmapHead(m.dim)
        self.decoder = QueryDecoder(cfg, self.pe)

    def forward(self, x, tokens, token_xyz, meta):
        h = self.cfg.model.heatmap
        P = self.backbone(x)
        G = self.adapter(tokens, token_xyz) if self.adapter is not None else None  # (B, 1728, D)
        hm, size = self.heatmap(P[h.level])
        ref_c, ref_s = self.heatmap.candidates(hm, size, h.level, h.n_q, h.nms_kernel)
        outs = self.decoder(P, G, ref_c, ref_s, meta, h.level)
        return {"hm": hm, "size": size, "layers": outs, "ref_center": ref_c, "G": G}
