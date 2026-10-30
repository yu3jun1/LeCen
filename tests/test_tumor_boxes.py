"""肿瘤框：Lung1 RTSTRUCT 的 GTV-1 外接框与 BAMF DICOM SEG 连通域框的世界坐标（合成数据）。"""
from types import SimpleNamespace as NS

import numpy as np
import pytest

import data.prep_acrin as prep_acrin
import data.prep_lung1 as prep_lung1


def seg_dataset():
    """行间距 2（y），列间距 3（x），层间距 5（z）；只存非空帧且乱序，k = 2、3 一个连通域，k = 6 另一个。"""
    rows, cols, origin = 10, 12, np.array([100.0, 200.0, -50.0])
    vol = np.zeros((7, rows, cols), np.uint8)
    vol[2:4, 4:7, 1:3] = 1
    vol[6, 0, 10] = 1
    frames, groups = [], []
    for k in (6, 2, 3):
        frames.append(vol[k])
        groups.append(NS(SegmentIdentificationSequence=[NS(ReferencedSegmentNumber=2)],
                         PlanePositionSequence=[NS(ImagePositionPatient=list(origin + [0, 0, 5 * k]))]))
    frames.append(np.ones((rows, cols), np.uint8))  # 另一个片段（肺）的帧不计入
    groups.append(NS(SegmentIdentificationSequence=[NS(ReferencedSegmentNumber=1)],
                     PlanePositionSequence=[NS(ImagePositionPatient=list(origin))]))
    shared = NS(PlaneOrientationSequence=[NS(ImageOrientationPatient=[1, 0, 0, 0, 1, 0])],
                PixelMeasuresSequence=[NS(PixelSpacing=[2.0, 3.0], SpacingBetweenSlices=5.0)])
    return NS(SegmentSequence=[NS(SegmentLabel="Lung", SegmentNumber=1), NS(SegmentLabel=prep_acrin.SEGMENT, SegmentNumber=2)],
              SharedFunctionalGroupsSequence=[shared], PerFrameFunctionalGroupsSequence=groups, pixel_array=np.stack(frames),
              NumberOfFrames=len(frames), Rows=rows, Columns=cols, FrameOfReferenceUID="for-1")


def test_seg_boxes(monkeypatch):
    monkeypatch.setattr(prep_acrin.pydicom, "dcmread", lambda path: seg_dataset())
    frame_of_reference, boxes = prep_acrin.seg_boxes("seg.dcm")
    assert frame_of_reference == "for-1" and len(boxes) == 2
    big = max(boxes, key=lambda b: b[3:].prod())
    assert np.allclose(big, [104.5, 210.0, -37.5, 6.0, 6.0, 10.0])
    small = min(boxes, key=lambda b: b[3:].prod())
    assert np.allclose(small, [130.0, 200.0, -20.0, 3.0, 2.0, 5.0])


def test_seg_source_priority():
    assert prep_acrin.source_of("AIMI lung and FDG tumor radiologist 1 corrected segmentation") == (0, "radiologist_corrected")
    assert prep_acrin.source_of("AIMI lung and FDG tumor non-expert 1 corrected segmentation") == (1, "nonexpert_corrected")
    assert prep_acrin.source_of("AIMI lung and FDG tumor AI segmentation") == (2, "ai")


def rtstruct(ref_uid, roi_name="GTV-1"):
    gtv = [NS(ContourData=[10, 20, -10, 30, 20, -10, 30, 50, -10]), NS(ContourData=[12, 22, -7, 28, 40, -7, 15, 45, -7])]
    cord = [NS(ContourData=[0, 0, -100, 1, 1, -100, 0, 1, -100])]
    return NS(ReferencedFrameOfReferenceSequence=[NS(RTReferencedStudySequence=[NS(RTReferencedSeriesSequence=[NS(SeriesInstanceUID=ref_uid)])])],
              StructureSetROISequence=[NS(ROINumber=1, ROIName=roi_name), NS(ROINumber=2, ROIName="Spinal-Cord")],
              ROIContourSequence=[NS(ReferencedROINumber=1, ContourSequence=gtv), NS(ReferencedROINumber=2, ContourSequence=cord)])


def test_gtv_box(tmp_path, monkeypatch):
    (tmp_path / "ct.dcm").touch()
    rt = {"ok": rtstruct("ct-uid"), "other": rtstruct("ct-uid", "GTV-2"), "wrong": rtstruct("another-uid")}

    def read(path, stop_before_pixels=False):
        return NS(SliceThickness=3.0) if stop_before_pixels else rt[path]
    monkeypatch.setattr(prep_lung1.pydicom, "dcmread", read)
    box = prep_lung1.gtv_box("ok", tmp_path, "ct-uid")
    assert np.allclose(box, [20.0, 35.0, -8.5, 20.0, 30.0, 6.0])  # z：两层轮廓 −10、−7 再加一个层厚
    assert prep_lung1.gtv_box("other", tmp_path, "ct-uid") is None
    with pytest.raises(ValueError):
        prep_lung1.gtv_box("wrong", tmp_path, "ct-uid")


def test_lung_only_seg_has_no_tumor_label_without_pixel_decode(monkeypatch):
    class LungOnly:
        FrameOfReferenceUID = "for-1"
        SegmentSequence = [NS(SegmentLabel="Lung", SegmentNumber=1)]

        @property
        def pixel_array(self):
            raise AssertionError("A lung-only SEG must not decode tumor pixels")

    monkeypatch.setattr(prep_acrin.pydicom, "dcmread", lambda path: LungOnly())
    assert prep_acrin.seg_boxes("lung.dcm") == ("for-1", [])
    assert prep_acrin.seg_boxes("lung.dcm", with_reason=True) == ("for-1", [], "no_tumor_label")


def test_empty_tumor_label_is_distinct_from_missing_label(monkeypatch):
    empty = seg_dataset()
    empty.pixel_array[:] = 0
    monkeypatch.setattr(prep_acrin.pydicom, "dcmread", lambda path: empty)
    assert prep_acrin.seg_boxes("empty.dcm", with_reason=True) == ("for-1", [], "empty_tumor_segment")


def test_acrin_prep_missing_labels_and_frame_identity(tmp_path, monkeypatch):
    """Real tumor geometry survives; absent SEG, lung-only, and wrong FoR stay distinct."""
    import json
    import sys
    import pandas as pd

    frozen_dir, prep_dir, bamf = [tmp_path / name for name in ("frozen", "prep", "bamf")]
    frozen_dir.mkdir()
    frozen = pd.DataFrame({"patient_id": ["none", "lung", "wrong", "tumor"],
                           "series_uid": ["ct-none", "ct-lung", "ct-wrong", "ct-tumor"],
                           "time_days": [10, 20, 30, 40], "event": [0, 1, 0, 1],
                           "age": [50, 51, 52, 53], "sex_male": [0, 1, 0, 1]})
    for name in ("IIIA", "IIIB", "Other"):
        frozen[f"baseline_stage_{name}"] = 0
        frozen[f"baseline_stage_{name}_missing"] = 0
    frozen.to_csv(frozen_dir / "clinical.csv", index=False)
    segs = pd.DataFrame({"PatientID": ["lung", "wrong", "tumor", "tumor"],
                         "SeriesInstanceUID": ["seg-lung", "seg-wrong", "seg-tumor", "seg-tumor-lung"],
                         "SeriesDescription": ["AI segmentation", "AI segmentation", "AI segmentation", "radiologist corrected segmentation"],
                         "collection_id": "acrin_nsclc_fdg_pet", "analysis_result_id": prep_acrin.ANALYSIS})
    for uid in segs.SeriesInstanceUID:
        path = bamf / uid
        path.mkdir(parents=True)
        (path / "seg.dcm").touch()
    lung = NS(FrameOfReferenceUID="for-1", SegmentSequence=[NS(SegmentLabel="Lung", SegmentNumber=1)])
    wrong = seg_dataset();wrong.FrameOfReferenceUID = "another-for"
    datasets = {"seg-lung": lung, "seg-wrong": wrong, "seg-tumor": seg_dataset(), "seg-tumor-lung": lung}
    monkeypatch.setattr(prep_acrin.pydicom, "dcmread", lambda path: datasets[path.parent.name])
    monkeypatch.setattr(prep_acrin, "ct_frame_of_reference", lambda raw, uid: "for-1")
    fake = NS(index=segs, download_from_selection=lambda **kwargs: None)
    monkeypatch.setitem(sys.modules, "idc_index", NS(IDCClient=lambda: fake))
    cfg = NS(clinical=NS(cohorts=NS(acrin=NS(prep=prep_dir, bamf=bamf,
        frozen_meta=frozen_dir, raw=tmp_path / "raw", collection="acrin_nsclc_fdg_pet"))))
    prep_acrin.main(cfg)

    clinical = pd.read_csv(prep_dir / "clinical.csv")
    series = pd.read_csv(prep_dir / "series.csv")
    assert clinical.patient_id.tolist() == frozen.patient_id.tolist()
    assert clinical.time_days.tolist() == frozen.time_days.tolist()
    assert clinical.event.tolist() == frozen.event.tolist()
    assert series.series_uid.tolist() == frozen.series_uid.tolist()
    missing = pd.read_csv(prep_dir / "tumors_unmatched.csv").set_index("patient_id")
    assert missing.reason.to_dict() == {"none": "no_seg", "lung": "no_tumor_label", "wrong": "frame_of_reference_differs"}
    assert json.loads(missing.loc["lung", "seg_uids"]) == ["seg-lung"]
    audit = pd.read_csv(prep_dir / "tumor_seg_audit.csv").set_index("seg_uid")
    assert audit.loc["seg-tumor-lung", "status"] == "no_tumor_label"
    assert audit.loc["seg-tumor", "status"] == "selected"
    assert audit.loc["seg-wrong", "status"] == "frame_of_reference_differs"
    assert not audit.loc["seg-wrong", "frame_of_reference_matches"]
    tumors = pd.read_csv(prep_dir / "tumors.csv")
    assert len(tumors) == 2 and set(tumors.patient_id) == {"tumor"}
    assert set(tumors.seg_uid) == {"seg-tumor"} and set(tumors.source) == {"ai"}
    boxes = [np.fromstring(value, sep=" ") for value in tumors.bbox_world]
    assert any(np.allclose(box, [104.5, 210.0, -37.5, 6.0, 6.0, 10.0]) for box in boxes)
    assert any(np.allclose(box, [130.0, 200.0, -20.0, 3.0, 2.0, 5.0]) for box in boxes)
