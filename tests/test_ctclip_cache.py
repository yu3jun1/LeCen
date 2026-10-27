"""缓存与在线提取一致（float16 精度内）；token 中心叠加到 CT 上位置合理（token 网格中心与 CT 物理中心重合）。

需要 GPU、CT-CLIP 代码与权重，以及仍保留的 subset 压缩包。
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from common.coords import affine_of, vox_to_world
from common.io import ROOT, load_config, read_image, read_npz
from data.cache_ctclip import extract, load_encoder


def test_cache_consistent(tmp_path):
    cfg = load_config(ROOT / "configs" / "data.yaml")
    folds = pd.read_csv(Path(cfg.paths.meta) / "folds.csv")
    k = int(folds.subset.min())
    shutil.unpack_archive(Path(cfg.paths.raw_luna16) / f"subset{k}.zip", tmp_path)
    encoder = load_encoder(cfg.ctclip.checkpoint)
    for sid in folds[folds.subset == k].seriesuid[:5]:
        img = read_image(tmp_path / f"subset{k}" / f"{sid}.mhd")
        tokens, xyz = extract(img, encoder, cfg.ctclip.token_layer, cfg.ctclip.pooled_grid)
        cached = read_npz(Path(cfg.paths.cache) / "ctclip" / f"{sid}.npz")
        assert np.abs(tokens.astype(np.float16).astype(np.float32) - cached["tokens"].astype(np.float32)).max() <= \
            np.abs(cached["tokens"].astype(np.float32)).max() * 2 ** -10
        assert np.allclose(xyz, cached["token_world_xyz"])
        corners = vox_to_world(np.array([[0, 0, 0], np.array(img.GetSize())[::-1] - 1], float), affine_of(img))
        assert np.abs(xyz.mean(0) - corners.mean(0)).max() < 30.0
