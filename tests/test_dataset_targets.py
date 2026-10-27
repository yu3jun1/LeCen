"""正例、截断、忽略区域、翻转后框、标签与 valid_mask 对应正确；越界区域 valid_mask = 0。"""
import numpy as np
import SimpleITK as sitk
from omegaconf import OmegaConf

from common.io import write_json, write_npz
from data.det_dataset import DetPatchDataset

CFG = OmegaConf.create({"patch_zyx": [80, 192, 192], "pos_jitter_mm": 10, "paths": {"cache": ""},
                        "augment": {"flip_axes": [0, 1, 2], "intensity": {"scale": 0.1, "shift": 0.05}}})


def make_case(tmp_path):
    vol = np.full((120, 260, 250), -800, np.int16)
    lab = np.zeros(vol.shape, np.uint16)
    lab[30:36, 100:110, 90:98] = 1          # A：完整落在 patch 内
    lab[60:66, 244:252, 120:128] = 2        # B：中心在 patch 外但框与 patch 相交
    img = sitk.GetImageFromArray(vol)
    img.SetSpacing((0.703125, 0.703125, 1.25))
    for d in ("vol", "lab", "tgt", "tok"):
        (tmp_path / d).mkdir()
    sitk.WriteImage(img, str(tmp_path / "vol" / "c.nii.gz"))
    l = sitk.GetImageFromArray(lab)
    l.CopyInformation(img)
    sitk.WriteImage(l, str(tmp_path / "lab" / "c.nii.gz"))
    inst = [{"center_vox": [32.5, 104.5, 93.5], "size_vox": [6, 10, 8], "has_mask": True, "label_value": 1},
            {"center_vox": [62.5, 247.5, 123.5], "size_vox": [6, 8, 8], "has_mask": True, "label_value": 2}]
    write_json(tmp_path / "tgt" / "c.json", {"instances": inst, "ignore": [{"center_vox": [50, 50, 50], "radius_vox": [4, 4, 4]},
                                                                           {"center_vox": [5, 5, 5], "radius_vox": [2, 2, 2]}]})
    write_npz(tmp_path / "tok" / "c.npz", tokens=np.zeros((1728, 512), np.float16), token_world_xyz=np.zeros((1728, 3), np.float32))
    cfg = CFG.copy()
    cfg.paths.cache = str(tmp_path)
    (tmp_path / "ctclip").symlink_to(tmp_path / "tok")
    return cfg


def test_targets(tmp_path):
    cfg = make_case(tmp_path)
    for augment, seed in ((False, 0), (True, 1), (True, 2), (True, 3)):
        ds = DetPatchDataset(cfg, ["c"], tmp_path / "vol", tmp_path / "lab", tmp_path / "tgt", augment)
        s = ds.patch_at(ds.case(0), np.array([60.0, 150.0, 100.0]), np.random.default_rng(seed))
        start, flip, p = s["offset_vox"].numpy(), s["flip"].numpy(), np.array(cfg.patch_zyx)
        assert (start == [20, 54, 4]).all()
        assert s["boxes"].shape == (1, 6) and s["label_value"].tolist() == [1]
        c = s["boxes"][0, :3].numpy()
        assert np.allclose(c, np.where(flip == 1, p - 1 - ([32.5, 104.5, 93.5] - start), [32.5, 104.5, 93.5] - start))
        z, y, x = np.round(c).astype(int)
        assert s["labels"][z, y, x] == 1
        # B 截断 → 忽略区域；ignore.csv 中 (50,50,50) 与 patch 相交 → 忽略区域；(5,5,5) 不相交
        assert s["ignore"].shape == (2, 6)
        assert s["valid_mask"].all()  # 该 patch 完整落在体积内
    ds = DetPatchDataset(cfg, ["c"], tmp_path / "vol", tmp_path / "lab", tmp_path / "tgt", True)
    s = ds.patch_at(ds.case(0), np.array([100.0, 230.0, 20.0]), np.random.default_rng(5))
    start, flip = s["offset_vox"].numpy(), s["flip"].numpy()
    valid = s["valid_mask"].numpy()
    v = np.flip(valid, tuple(np.flatnonzero(flip)))  # 撤销翻转后与体积范围比较
    zz, yy, xx = np.meshgrid(*[np.arange(n) + o for n, o in zip(cfg.patch_zyx, start)], indexing="ij")
    inside = (zz >= 0) & (zz < 120) & (yy >= 0) & (yy < 260) & (xx >= 0) & (xx < 250)
    assert (v == inside).all() and not v.all()
    img = np.flip(s["image"][0].numpy(), tuple(np.flatnonzero(flip)))
    assert (img[~inside] == 0).all()
