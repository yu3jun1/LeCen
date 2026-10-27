"""临床数据预处理。

python -m data.prep_clinical --cfg configs/data.yaml

1. idc-index 在 collection 的 CT 序列中先汇总 series_size_MB，只下载资格核验确定的治疗前基线 CT。
2. 读取 DICOM 序列 → 完整体积上提取 CT-CLIP token → lungmask 肺分割 → 临床裁剪规则（必要时回退）→ 重采样保存（同 LUNA16）。
3. 肿瘤保留率只做事后审计（crop_rules --audit），不据此修改任何病例的裁剪。
4. 清洗临床表：统一时间起点、事件编码、治疗编码（只用预测时已知的方案）、缺失指示。
肿瘤框来自 meta/clinical_tumors.csv（patient_id, tumor_id, center_world_xyz, bbox_world[cccwhd], source）。
"""
import argparse
from pathlib import Path

import pandas as pd
import SimpleITK as sitk

from common.coords import affine_of, world_to_vox
from common.io import load_config, upsert_csv, world_list, write_csv, write_image, write_json
from data.cache_ctclip import cache, load_encoder
from data.crop_rules import spacing_zyx
from data.prep_luna16 import preprocess


def clean_table(cfg):
    """time_days 为相对统一起点的天数；event 为 0/1；数值协变量保留原值，类别协变量 one-hot；均附缺失指示。"""
    c = cfg.clinical
    raw = pd.read_csv(cfg.paths.clinical_table, dtype={c.columns.patient_id: str})
    df = pd.DataFrame({"patient_id": raw[c.columns.patient_id],
                       "time_days": raw[c.columns.time_days].astype(float),
                       "event": raw[c.columns.event].isin(list(c.event_values)).astype(int),
                       "treatment": raw[c.columns.treatment].astype(str)})
    for v in c.covariates:
        x = raw[v]
        cols = x.to_frame() if pd.api.types.is_numeric_dtype(x) else pd.get_dummies(x, prefix=v, dtype=float)
        df = pd.concat([df, cols, x.isna().astype(int).rename(f"{v}_missing")], axis=1)
    return df


def download(cfg):
    from idc_index import IDCClient

    client = IDCClient()
    series = pd.read_csv(cfg.paths.clinical_series, dtype={"patient_id": str})
    idx = client.index
    ct = idx[(idx.collection_id == cfg.clinical.collection) & (idx.Modality == "CT")]
    base = ct[ct.SeriesInstanceUID.isin(series.series_uid)]
    print(f"{cfg.clinical.collection} CT: {len(ct)} 序列 {ct.series_size_MB.sum():.0f} MB；"
          f"基线: {len(base)} 序列 {base.series_size_MB.sum():.0f} MB")
    client.download_from_selection(downloadDir=cfg.paths.raw_clinical, seriesInstanceUID=list(series.series_uid),
                                   dirTemplate="%SeriesInstanceUID")
    return series


def read_series(d):
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(d)))
    return reader.Execute()


def write_targets(path, img, boxes):
    """肿瘤框（世界 cccwhd）→ 预处理体积体素坐标；临床标注没有 mask。"""
    sp = spacing_zyx(img)
    inst = [{"center_vox": world_to_vox(b[:3], affine_of(img))[0].tolist(), "size_vox": (b[[5, 4, 3]] / sp).tolist(),
             "has_mask": False, "label_value": 0} for b in boxes]
    write_json(path, {"instances": inst, "ignore": []})


def main(cfg):
    from lungmask import LMInferer

    p = cfg.paths
    proc, meta = Path(p.processed), Path(p.meta)
    series = download(cfg)
    tumors = pd.read_csv(meta / "clinical_tumors.csv", dtype={"patient_id": str})
    encoder, lm = load_encoder(cfg.ctclip.checkpoint), LMInferer()
    crops = []
    for r in series.itertuples():
        img = read_series(Path(p.raw_clinical) / r.series_uid)
        cache(Path(p.cache) / "ctclip" / f"{r.patient_id}.npz", img, encoder, cfg)
        out, rec = preprocess(img, lm.apply(img), cfg.preprocess.crop.clinical, cfg)
        write_image(out, proc / "clinical" / f"{r.patient_id}.nii.gz")
        write_targets(proc / "clinical_targets" / f"{r.patient_id}.json", out,
                      world_list(tumors[tumors.patient_id == r.patient_id].bbox_world))
        crops.append({"case_id": r.patient_id, "dataset": "clinical", **rec})
        print(r.patient_id, rec)
    upsert_csv(meta / "crops.csv", pd.DataFrame(crops), "case_id")
    write_csv(meta / "clinical.csv", clean_table(cfg))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    main(load_config(ap.parse_args().cfg))
