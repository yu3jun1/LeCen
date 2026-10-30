"""ACRIN 6668 外部测试队列：写出统一格式的 clinical.csv、series.csv、tumors.csv。

python -m data.prep_acrin --cfg configs/data.yaml

clinical.csv / series.csv：来自已冻结的 115 例 ACRIN 模型输入（cohorts.acrin.frozen_meta/clinical.csv，含审计后的 OS 与基线 CT）。
  与 Lung1 对齐：time_days、event 原值；age、sex_male 原值；stage 由 baseline_stage_IIIA / IIIB / Other 还原，
  三者全 0 为参照 IIB（记为 II），任一缺失为缺失；baseline_PS 不进入（Lung1 没有此变量）；
  治疗统一为 RT±chemo（Lung1 没有治疗细分，ACRIN 全为计划 CRT）。
tumors.csv：IDC 中 BAMF-AIMI 的 DICOM SEG（"FDG-Avid Tumor" 片段），每个 3D 连通域一个世界坐标框。
  SEG 定义在 PET 网格上：只采用 FrameOfReferenceUID 与所选基线 CT 严格相同的 SEG（共坐标系；不等同于同次扫描），
  缺少 FDG-Avid Tumor 的肺分割不产生肿瘤框。逐 SEG 的来源、UID 与拒绝理由记入 tumor_seg_audit.csv。
  同一患者在非空且坐标系匹配的候选中按 radiologist_corrected > nonexpert_corrected > ai 取一个；其余患者记入 tumors_unmatched.csv。
  这些框只用于报告外部队列的主肿瘤覆盖率（recall chain），不参与训练、slot 选择或任何阈值确定。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from scipy import ndimage

from common.io import load_config, vec_str, write_csv
from data.prep_clinical import series_directory
from data.prep_lung1 import TREATMENT

ANALYSIS = "bamf_aimi_annotations"
SEGMENT = "FDG-Avid Tumor"
SOURCES = [("radiologist", "radiologist_corrected"), ("non-expert", "nonexpert_corrected"), ("", "ai")]  # 按优先级


def harmonize(frozen):
    stage = pd.Series("II", index=frozen.index, dtype=object)
    for col, name in (("baseline_stage_IIIA", "IIIA"), ("baseline_stage_IIIB", "IIIB"), ("baseline_stage_Other", "other")):
        stage[frozen[col] == 1] = name
        stage[frozen[f"{col}_missing"] == 1] = np.nan
    return pd.DataFrame({"patient_id": frozen.patient_id, "time_days": frozen.time_days, "event": frozen.event,
                         "treatment": TREATMENT, "age": frozen.age, "sex_male": frozen.sex_male, "stage": stage})


def source_of(description):
    return next((rank, name) for rank, (key, name) in enumerate(SOURCES) if key in description)


def seg_boxes(path, label=SEGMENT, *, with_reason=False):
    """片段连通域 → 世界 cccwhd 框；with_reason 附带无框原因。

    原两项返回接口保留。肺分割无肿瘤标签时在访问几何或解码像素前返回，
    与有肿瘤标签但没有前景体素的 empty_tumor_segment 明确区分。
    """
    d = pydicom.dcmread(path)

    def result(boxes, reason=""):
        value = (d.FrameOfReferenceUID, boxes)
        return (*value, reason) if with_reason else value

    seg = {s.SegmentLabel: s.SegmentNumber for s in d.SegmentSequence}.get(label)
    if seg is None:
        return result([], "no_tumor_label")
    sel = [(i, np.array(f.PlanePositionSequence[0].ImagePositionPatient, float)) for i, f in enumerate(d.PerFrameFunctionalGroupsSequence)
           if f.SegmentIdentificationSequence[0].ReferencedSegmentNumber == seg]
    if not sel:
        return result([], "empty_tumor_segment")
    sh = d.SharedFunctionalGroupsSequence[0]
    iop = np.array(sh.PlaneOrientationSequence[0].ImageOrientationPatient, float)
    row_dir, col_dir = iop[:3], iop[3:]          # 沿列号增大 / 行号增大的方向
    normal = np.cross(row_dir, col_dir)
    sp_row, sp_col = map(float, sh.PixelMeasuresSequence[0].PixelSpacing)
    dz = float(sh.PixelMeasuresSequence[0].SpacingBetweenSlices)
    frames = np.asarray(d.pixel_array).reshape(int(d.NumberOfFrames), d.Rows, d.Columns)
    pos = np.array([p for _, p in sel])
    origin = pos[np.argmin(pos @ normal)]
    k = np.round((pos - origin) @ normal / dz).astype(int)
    vol = np.zeros((k.max() + 1, d.Rows, d.Columns), bool)
    for (i, _), kk in zip(sel, k):
        vol[kk] |= frames[i] > 0
    lab, _ = ndimage.label(vol)
    boxes = []
    for sl in ndimage.find_objects(lab):
        idx = np.array(np.meshgrid(*[[s.start - 0.5, s.stop - 0.5] for s in sl], indexing="ij")).reshape(3, -1).T
        world = origin + idx[:, :1] * dz * normal + idx[:, 1:2] * sp_row * col_dir + idx[:, 2:] * sp_col * row_dir
        lo, hi = world.min(0), world.max(0)
        boxes.append(np.r_[(lo + hi) / 2, hi - lo])
    return result(boxes, "" if boxes else "empty_tumor_segment")


def ct_frame_of_reference(raw, series_uid):
    f = next(series_directory(raw, series_uid).glob("*.dcm"))
    return pydicom.dcmread(f, stop_before_pixels=True).FrameOfReferenceUID


def main(cfg):
    from idc_index import IDCClient

    c = cfg.clinical.cohorts.acrin
    prep, bamf = Path(c.prep), Path(c.bamf)
    frozen = pd.read_csv(Path(c.frozen_meta) / "clinical.csv", dtype={"patient_id": str})
    write_csv(prep / "clinical.csv", harmonize(frozen))
    write_csv(prep / "series.csv", frozen[["patient_id", "series_uid"]])

    client = IDCClient()
    idx = client.index
    segs = idx[(idx.collection_id == c.collection) & (idx.analysis_result_id == ANALYSIS) & idx.PatientID.isin(frozen.patient_id)]
    client.download_from_selection(downloadDir=bamf, seriesInstanceUID=list(segs.SeriesInstanceUID), dirTemplate="%SeriesInstanceUID", use_s5cmd_sync=True)
    tumors, unmatched, seg_audit = [], [], []
    for r in frozen.itertuples():
        mine = segs[segs.PatientID == r.patient_id]
        if mine.empty:
            unmatched.append({"patient_id": r.patient_id, "n_seg": 0, "reason": "no_seg", "seg_uids": "[]", "sources": "[]"})
            continue
        ct_for = ct_frame_of_reference(c.raw, r.series_uid)
        found, patient_audit = [], []
        for s in mine.itertuples():
            rank, source = source_of(s.SeriesDescription)
            seg_for, boxes, reason = seg_boxes(next((bamf / s.SeriesInstanceUID).glob("*.dcm")), with_reason=True)
            # Missing tumor labels are not a negative tumor annotation. Every
            # actual tumor candidate must still pass the exact FoR check.
            if reason != "no_tumor_label" and seg_for != ct_for:
                reason = "frame_of_reference_differs"
            record = {"patient_id": r.patient_id, "seg_uid": s.SeriesInstanceUID, "source": source,
                      "ct_frame_of_reference_uid": ct_for, "seg_frame_of_reference_uid": seg_for,
                      "frame_of_reference_matches": seg_for == ct_for, "n_boxes": len(boxes),
                      "status": reason or "eligible_candidate"}
            patient_audit.append(record)
            if not reason:
                found.append((rank, source, s.SeriesInstanceUID, boxes))
        seg_audit.extend(patient_audit)
        if not found:
            reasons = {record["status"] for record in patient_audit}
            reason = next(value for value in ("empty_tumor_segment", "frame_of_reference_differs", "no_tumor_label") if value in reasons)
            unmatched.append({"patient_id": r.patient_id, "n_seg": len(mine), "reason": reason,
                              "seg_uids": json.dumps([record["seg_uid"] for record in patient_audit]),
                              "sources": json.dumps([record["source"] for record in patient_audit])})
            continue
        _, source, seg_uid, boxes = min(found, key=lambda x: (x[0], x[2]))
        for record in patient_audit:
            if record["status"] == "eligible_candidate":
                record["status"] = "selected" if record["seg_uid"] == seg_uid else "not_selected_lower_priority"
        tumors += [{"patient_id": r.patient_id, "tumor_id": i, "bbox_world": vec_str(b), "source": source, "seg_uid": seg_uid}
                   for i, b in enumerate(boxes)]
    write_csv(prep / "tumors.csv", pd.DataFrame(tumors, columns=["patient_id", "tumor_id", "bbox_world", "source", "seg_uid"]))
    write_csv(prep / "tumors_unmatched.csv", pd.DataFrame(unmatched, columns=["patient_id", "n_seg", "reason", "seg_uids", "sources"]))
    write_csv(prep / "tumor_seg_audit.csv", pd.DataFrame(seg_audit, columns=["patient_id", "seg_uid", "source", "ct_frame_of_reference_uid", "seg_frame_of_reference_uid", "frame_of_reference_matches", "n_boxes", "status"]))
    print(f"ACRIN {len(frozen)} 例；有可用 BAMF 框 {len({t['patient_id'] for t in tumors})} 例；",
          pd.DataFrame(unmatched, columns=["reason"]).reason.value_counts().to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    main(load_config(ap.parse_args().cfg))
