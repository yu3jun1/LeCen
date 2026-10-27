"""LUNA16 结节 ↔ pylidc 标注簇对应表、训练框、targets json 与忽略区域。

匹配：每个 seriesuid 内，LUNA 结节中心与簇中心（共识 mask 质心）的欧氏距离矩阵上做匈牙利一一匹配，
接受阈值 = max(0.5 × LUNA 直径, 3 mm)。
状态：one2one；one2many（结节附近有多个簇，取最近者并记录其余）；many2one（一个簇对应多个结节）；unmatched。
unmatched 与 many2one 的结节 has_mask = false，训练框为直径立方体，进入人工核查列表。

python -m data.build_instances --cfg configs/data.yaml   # 汇总对应表
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from common.coords import affine_of, world_to_vox
from common.io import load_config, upsert_csv, vec_str, world_list, write_csv, write_json

XYZ = ["coordX", "coordY", "coordZ"]


def write_ignore(cfg):
    """annotations_excluded → ignore.csv；直径无效（< 0）时按官方脚本取 10 mm。"""
    ex = pd.read_csv(cfg.paths.luna16_excluded)
    d = ex.diameter_mm.where(ex.diameter_mm >= 0, 10.0)
    write_csv(Path(cfg.paths.meta) / "ignore.csv", pd.DataFrame({
        "seriesuid": ex.seriesuid, "center_world_xyz": [vec_str(v) for v in ex[XYZ].to_numpy()], "diameter_mm": d}))


def match(luna_xyz, luna_d, cl_xyz):
    """返回每个 LUNA 结节的 (状态, 簇下标或 −1, 其余近邻簇下标)。"""
    if len(cl_xyz) == 0:
        return [("unmatched", -1, [])] * len(luna_xyz)
    dist = cdist(luna_xyz, cl_xyz)
    near = dist < np.maximum(0.5 * luna_d, 3.0)[:, None]
    row, col = linear_sum_assignment(np.where(near, dist, 1e6))
    assigned = {r: c for r, c in zip(row, col) if near[r, c]}
    out = []
    for i in range(len(luna_xyz)):
        js = np.flatnonzero(near[i])
        if len(js) == 0:
            out.append(("unmatched", -1, []))
        elif (near[:, js].sum(0) > 1).any():
            out.append(("many2one", int(js[np.argmin(dist[i, js])]), []))
        else:
            j = assigned[i]
            out.append(("one2many" if len(js) > 1 else "one2one", int(j), [int(x) for x in js if x != j]))
    return out


def update(cfg, sid, label_img, clusters, patient_id):
    """更新 sid 的对应表行、核查列表，并写 processed/luna16_targets/<sid>.json（预处理体积体素坐标）。"""
    meta = Path(cfg.paths.meta)
    ann = pd.read_csv(cfg.paths.luna16_annotations)
    ann = ann[ann.seriesuid == sid]
    ign = pd.read_csv(meta / "ignore.csv")
    ign = ign[ign.seriesuid == sid]
    a = affine_of(label_img)
    sp = np.array(label_img.GetSpacing()[::-1])
    label = sitk.GetArrayFromImage(label_img)
    cl_xyz = np.array([c["center_world_xyz"] for c in clusters]).reshape(-1, 3)
    rows, review, targets = [], [], []
    for (idx, r), (status, j, others) in zip(ann.iterrows(), match(ann[XYZ].to_numpy(), ann.diameter_mm.to_numpy(), cl_xyz)):
        has_mask = status in ("one2one", "one2many")
        lv = clusters[j]["label_value"] if has_mask else 0
        if has_mask:
            vox = np.argwhere(label == lv)
            lo, hi = vox.min(0), vox.max(0) + 1
            center, size = (lo + hi - 1) / 2, (hi - lo).astype(float)
        else:
            center, size = world_to_vox(r[XYZ].to_numpy(float), a)[0], r.diameter_mm / sp
        rows.append({"inst_id": f"{sid}_{idx}", "seriesuid": sid, "patient_id": patient_id, "luna_idx": idx,
                     "pylidc_cluster_id": clusters[j]["pylidc_cluster_id"] if j >= 0 else -1,
                     "center_world_xyz": vec_str(r[XYZ].to_numpy(float)), "diameter_mm": r.diameter_mm,
                     "has_mask": has_mask, "label_value": lv, "train_box_source": "mask" if has_mask else "diameter",
                     "match_status": status})
        if status != "one2one":
            review.append({"inst_id": f"{sid}_{idx}", "seriesuid": sid, "match_status": status,
                           "other_cluster_ids": " ".join(str(clusters[o]["pylidc_cluster_id"]) for o in others)})
        targets.append({"center_vox": center.tolist(), "size_vox": size.tolist(), "has_mask": has_mask, "label_value": int(lv)})
    ignore = [{"center_vox": world_to_vox(c, a)[0].tolist(), "radius_vox": (d / 2 / sp).tolist()}
              for c, d in zip(world_list(ign.center_world_xyz), ign.diameter_mm)]
    write_json(Path(cfg.paths.processed) / "luna16_targets" / f"{sid}.json", {"instances": targets, "ignore": ignore})
    if rows:
        upsert_csv(meta / "instances.csv", pd.DataFrame(rows), "seriesuid")
    if review:
        upsert_csv(meta / "instances_review.csv", pd.DataFrame(review), "seriesuid")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    cfg = load_config(ap.parse_args().cfg)
    meta = Path(cfg.paths.meta)
    inst = pd.read_csv(meta / "instances.csv")
    print(inst.match_status.value_counts().to_string())
    print(f"has_mask: {int(inst.has_mask.sum())} / {len(inst)}")
    print(f"人工核查列表：{meta / 'instances_review.csv'}（{len(pd.read_csv(meta / 'instances_review.csv'))} 条）")
