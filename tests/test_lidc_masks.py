"""抽查结节：LUNA16 中心落在映射后的 mask 内或距离 < 半径（叠加图由 tools/visualize.py 生成后人工确认）。"""
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from common.coords import affine_of, vox_to_world, world_to_vox
from common.io import ROOT, load_config, read_image, world_list


def test_luna_center_in_mask():
    cfg = load_config(ROOT / "configs" / "data.yaml")
    inst = pd.read_csv(Path(cfg.paths.meta) / "instances.csv")
    inst = inst[inst.has_mask].sample(30, random_state=0)
    for r in inst.itertuples():
        img = read_image(Path(cfg.paths.processed) / "luna16_labels" / f"{r.seriesuid}.nii.gz")
        lab, a = sitk.GetArrayFromImage(img), affine_of(img)
        c = world_list([r.center_world_xyz])[0]
        z, y, x = np.round(world_to_vox(c, a)[0]).astype(int)
        mask_world = vox_to_world(np.argwhere(lab == r.label_value).astype(float), a)
        dist = np.linalg.norm(mask_world - c, axis=1).min()
        assert lab[z, y, x] == r.label_value or dist < r.diameter_mm / 2, r.inst_id
