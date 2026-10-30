"""NSCLC-Radiomics（Lung1，MAASTRO）开发队列：写出统一格式的 clinical.csv、series.csv、tumors.csv。

python -m data.prep_lung1 --cfg configs/data.yaml

1. idc-index 下载 collection 的全部 CT 与 RTSTRUCT 到 cohorts.lung1.raw（patient_id/study_uid/series_uid/）。
   每位患者一个 CT 序列与一个 RTSTRUCT，RTSTRUCT 必须引用该 CT 序列。
2. 临床表来自 idc-index 的 nsclc_radiomics_clinical：Survival.time（天）、deadstatus.event（1 死亡 / 0 删失）、
   age（NA 为缺失）、gender、Overall.Stage（I / II / IIIa / IIIb → I / II / IIIA / IIIB，NA 为缺失）。
   Lung1 为根治性放疗或放化疗，没有治疗细分：treatment 统一为 RT±chemo（与外部队列一致）。
3. 主肿瘤框：ROI "GTV-1" 全部轮廓点的外接框，z 方向加一个 CT 层厚（每条轮廓代表一层）；
   没有 GTV-1 的患者记入 tumors_missing.csv，不进入 tumors.csv（不影响生存标签）。
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

from common.io import load_config, vec_str, write_csv
from data.prep_clinical import series_directory

TREATMENT = "RT±chemo"
STAGE = {"I": "I", "II": "II", "IIIa": "IIIA", "IIIb": "IIIB"}


def download(client, collection, raw):
    idx = client.index
    sel = idx[(idx.collection_id == collection) & idx.Modality.isin(["CT", "RTSTRUCT"])]
    print(f"{collection}: {len(sel)} 序列 {sel.series_size_MB.sum():.0f} MB")
    client.download_from_selection(downloadDir=raw, seriesInstanceUID=list(sel.SeriesInstanceUID),
                                   dirTemplate="%PatientID/%StudyInstanceUID/%SeriesInstanceUID", use_s5cmd_sync=True)
    return sel


def one_series(sel, modality):
    s = sel[sel.Modality == modality]
    if s.PatientID.duplicated().any():
        raise ValueError(f"{modality} 不是一患者一个序列：{sorted(set(s.PatientID[s.PatientID.duplicated()]))}")
    return s.set_index("PatientID").SeriesInstanceUID


def clinical_table(client, source_path=None):
    client.fetch_index("clinical_index")
    t = client.get_clinical_table("nsclc_radiomics_clinical")
    if source_path is not None:
        write_csv(source_path, t)
    return pd.DataFrame({"patient_id": t.patientid, "time_days": t.survivaltime.astype(float), "event": t.deadstatusevent,
                         "treatment": TREATMENT, "age": pd.to_numeric(t.age, errors="coerce"),
                         "sex_male": (t.gender == "male").astype(float), "stage": t.overallstage.map(STAGE)})


def gtv_box(rt_path, ct_dir, ct_uid, roi="gtv-1"):
    """RTSTRUCT 中 ROI 的世界 cccwhd 框；没有该 ROI 或 ROI 没有轮廓时返回 None。"""
    rt = pydicom.dcmread(rt_path)
    ref = rt.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID
    if ref != ct_uid:
        raise ValueError(f"{rt_path} 引用的序列 {ref} 不是该患者的 CT {ct_uid}")
    names = {r.ROINumber: r.ROIName.strip().lower() for r in rt.StructureSetROISequence}
    pts = [np.array(s.ContourData, float).reshape(-1, 3) for c in rt.ROIContourSequence
           if names[c.ReferencedROINumber] == roi for s in getattr(c, "ContourSequence", [])]
    if not pts:
        return None
    pts = np.concatenate(pts)
    thickness = float(pydicom.dcmread(next(Path(ct_dir).glob("*.dcm")), stop_before_pixels=True).SliceThickness)
    lo, hi = pts.min(0), pts.max(0)
    return np.r_[(lo + hi) / 2, hi - lo + [0, 0, thickness]]


def main(cfg):
    from idc_index import IDCClient

    c = cfg.clinical.cohorts.lung1
    prep = Path(c.prep)
    client = IDCClient()
    sel = download(client, c.collection, c.raw)
    write_csv(prep / "idc_download_selection.csv", sel)
    ct, rt = one_series(sel, "CT"), one_series(sel, "RTSTRUCT")
    write_csv(prep / "clinical.csv", clinical_table(client, prep / "idc_clinical_source.csv"))
    write_csv(prep / "series.csv", pd.DataFrame({"patient_id": ct.index, "series_uid": ct.to_numpy()}))
    tumors, missing = [], []
    for pid, ct_uid in ct.items():
        if pid not in rt:
            missing.append({"patient_id": pid, "reason": "no_rtstruct"})
            continue
        ct_dir = series_directory(c.raw, ct_uid)
        box = gtv_box(next(series_directory(c.raw, rt[pid]).glob("*.dcm")), ct_dir, ct_uid)
        if box is None:
            missing.append({"patient_id": pid, "reason": "no_gtv-1"})
            continue
        tumors.append({"patient_id": pid, "tumor_id": 0, "bbox_world": vec_str(box), "source": "rtstruct_gtv-1"})
    write_csv(prep / "tumors.csv", pd.DataFrame(tumors, columns=["patient_id", "tumor_id", "bbox_world", "source"]))
    write_csv(prep / "tumors_missing.csv", pd.DataFrame(missing, columns=["patient_id", "reason"]))
    print(f"CT {len(ct)}，GTV-1 框 {len(tumors)}，缺失 {len(missing)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    main(load_config(ap.parse_args().cfg))
