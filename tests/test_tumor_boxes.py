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
