"""Probability ties preserve strict p_min and inclusive FROC boundary semantics."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from engine import infer_det
from eval import recall_chain
from eval.froc import froc


def stage(scores):
    scores = np.asarray(scores, dtype=np.float32)
    boxes = np.zeros((len(scores), 6), dtype=np.float32)
    boxes[:, 3:] = 2
    return {'dec_p': scores, 'dec_box': boxes, 'keep': np.arange(len(scores)),
            'hm_xyz': np.zeros((1, 3), dtype=np.float32)}


def test_float64_predecessor_is_not_rounded_back_to_float32_score():
    score = np.float32(0.1)
    threshold = np.nextafter(np.float64(score), -np.inf)
    scores = np.array([score, score], dtype=np.float32)
    assert infer_det.score_threshold_mask(scores, threshold).tolist() == [True, True]
    assert infer_det.score_threshold_mask(scores, np.float64(score)).tolist() == [False, False]
    assert len(infer_det.preds_frame({'scan': stage(scores)}, threshold)) == 2


def test_bfloat16_repeated_scores_keep_exact_boundary_semantics():
    scores = torch.tensor([0.1, 0.1001, 0.2], dtype=torch.bfloat16).float().numpy()
    assert scores[0] == scores[1]
    threshold = float(scores[0])
    assert infer_det.score_threshold_mask(scores, threshold).tolist() == [False, False, True]
    assert infer_det.score_threshold_mask(scores, threshold, inclusive=True).tolist() == [True, True, True]
    predecessor = np.nextafter(np.float64(threshold), -np.inf)
    assert len(infer_det.preds_frame({'scan': stage(scores)}, predecessor)) == 3


@pytest.mark.parametrize('quantized', [False, True])
def test_fit_eight_fp_includes_the_eighth_score_and_all_ties(monkeypatch, quantized):
    boundary = np.float32(0.1)
    if quantized:
        boundary = torch.tensor(0.1, dtype=torch.bfloat16).float().numpy()
    scores = np.r_[np.full(7, 0.5, dtype=np.float32), np.full(3, boundary, dtype=np.float32), np.float32(0.01)]
    stages = {'scan': stage(scores)}

    def false_positive_records(cfg, predictions, sids):
        return {'scan': {'fp': predictions.probability.to_numpy(float)}}

    monkeypatch.setattr(infer_det, 'scan_records', false_positive_records)
    threshold = infer_det.fit_p_min(stages, None, 0.2, target_fp=8)
    assert threshold == np.nextafter(np.float64(boundary), -np.inf)
    selected = infer_det.preds_frame(stages, threshold)
    assert len(selected) == 10
    assert (selected.probability == boundary).sum() == 3
    assert infer_det.fit_p_min(stages, None, threshold, target_fp=8) == threshold


def test_fit_preserves_existing_threshold_when_it_already_reaches_eight_fp(monkeypatch):
    stages = {'scan': stage(np.full(8, 0.5, dtype=np.float32))}
    monkeypatch.setattr(infer_det, 'scan_records', lambda cfg, predictions, sids: {
        'scan': {'fp': predictions.probability.to_numpy(float)}})
    assert infer_det.fit_p_min(stages, None, 0.2, target_fp=8) == 0.2


def test_froc_and_recall_chain_include_the_same_float32_tied_boundary():
    score = float(np.float32(0.1))
    records = {'scan': {'nod_p': np.array([score]), 'nod_d': np.array([4.0]), 'fp': np.array([0.2])}}
    threshold = recall_chain.work_threshold(records, target_fp=1)
    assert threshold == score
    fps, sensitivity, thresholds = froc(list(records.values()))
    assert sensitivity[np.flatnonzero(thresholds == threshold)[0]] == 1
    output = recall_chain.chain(stage([score, score]), True, lambda xyz: np.ones(len(xyz), bool),
                                threshold, inclusive=True)
    assert output['work']
    # Clinical p_min intentionally remains a strict probability filter.
    strict = recall_chain.chain(stage([score, score]), True, lambda xyz: np.ones(len(xyz), bool), threshold)
    assert not strict['work']


def test_luna_chain_passes_the_exact_inclusive_froc_work_threshold(monkeypatch):
    score = float(np.float32(0.1))
    records = {'scan': {'nod_p': np.array([score]), 'nod_d': np.array([4.0]), 'fp': np.array([0.2])}}
    annotations = pd.DataFrame([{'seriesuid': 'scan', 'coordX': 0., 'coordY': 0., 'coordZ': 0., 'diameter_mm': 4.}])
    monkeypatch.setattr(recall_chain.pd, 'read_csv', lambda path: annotations)
    monkeypatch.setattr(recall_chain, 'read_header', lambda path: SimpleNamespace(GetSize=lambda: (32, 32, 32)))
    monkeypatch.setattr(recall_chain, 'affine_of', lambda header: np.eye(4))
    monkeypatch.setattr(recall_chain, 'spacing_zyx', lambda header: np.ones(3))
    monkeypatch.setattr(recall_chain, 'world_to_vox', lambda xyz, affine: np.asarray(xyz)[None])
    monkeypatch.setattr(recall_chain, 'retained', lambda *args: True)
    cfg = SimpleNamespace(paths=SimpleNamespace(luna16_annotations='unused.csv', processed='unused'))
    result = recall_chain.luna_chain(cfg, {'scan': stage([score, score])}, records)
    assert len(result) == 1 and result.work.all()


@pytest.mark.parametrize('quantized', [False, True])
@pytest.mark.parametrize('include_ties', [False, True])
def test_slots_predictions_and_clinical_chain_share_strict_threshold_mask(tmp_path, monkeypatch, quantized, include_ties):
    from engine import extract_slots

    score = np.float32(0.1)
    if quantized:
        score = torch.tensor(0.1, dtype=torch.bfloat16).float().numpy()
    threshold = np.nextafter(np.float64(score), -np.inf) if include_ties else float(score)
    st = stage([score, score, 0.2])
    st['dec_box'][-1, :3] = 20  # Only the boundary-score candidates cover the tumor.
    st['dec_h'] = np.arange(3, dtype=np.float32)[:, None]
    tables = {
        'splits_clin.csv': pd.DataFrame([{'patient_id': 'patient', 'split': 'val'}]),
        'clinical.csv': pd.DataFrame([{'patient_id': 'patient', 'series_uid': 'series'}]),
        'clinical_tumors.csv': pd.DataFrame([{'patient_id': 'patient', 'bbox_world': '0 0 0 2 2 2'}]),
        'crops.csv': pd.DataFrame([{'case_id': 'patient', 'crop_mode': 'body', 'fallback_reason': ''}]),
    }
    cfg = SimpleNamespace(paths=SimpleNamespace(cache=str(tmp_path/'cache'), processed=str(tmp_path/'processed'), meta=str(tmp_path/'meta')),
                          preprocess=SimpleNamespace(version='synthetic'), patch_zyx=[32,32,32], infer=SimpleNamespace(overlap=0.25))
    monkeypatch.setattr(extract_slots.pd, 'read_csv', lambda path, **kwargs: tables[Path(path).name].copy())
    monkeypatch.setattr(extract_slots, 'load_detector', lambda run, device: (cfg, None))
    monkeypatch.setattr(extract_slots, 'read_json', lambda path: {'p_min': threshold})
    monkeypatch.setattr(extract_slots, 'extractor_hash', lambda *args: 'synthetic-extractor')
    monkeypatch.setattr(extract_slots, 'file_sha1', lambda path: 'synthetic-input-hash')
    monkeypatch.setattr(extract_slots, 'InferCase', lambda *args: None)
    monkeypatch.setattr(extract_slots, 'infer_case', lambda *args: st)
    captured = {}
    monkeypatch.setattr(extract_slots, 'write_npz', lambda path, **arrays: captured.update(arrays))
    extract_slots.extract(tmp_path/'run', 'cpu')
    predictions = infer_det.preds_frame({'patient': st}, threshold)
    assert np.array_equal(captured['p'], predictions.probability.to_numpy())
    assert len(captured['p']) == (3 if include_ties else 1)
    monkeypatch.setattr(recall_chain, 'read_header', lambda path: SimpleNamespace(GetSize=lambda: (32,32,32)))
    monkeypatch.setattr(recall_chain, 'affine_of', lambda header: np.eye(4))
    monkeypatch.setattr(recall_chain, 'spacing_zyx', lambda header: np.ones(3))
    monkeypatch.setattr(recall_chain, 'world_to_vox', lambda xyz, affine: np.asarray(xyz)[None])
    monkeypatch.setattr(recall_chain, 'retained', lambda *args: True)
    rows, summary = recall_chain.clinical_chain(cfg, {'patient': st}, threshold, margin_mm=0)
    assert bool(rows.work.iloc[0]) == include_ties
