"""External patients without aligned BAMF boxes remain valid unlabeled inputs."""
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import SimpleITK as sitk
from omegaconf import OmegaConf

from data import crop_rules
from eval import recall_chain


def test_external_chain_with_no_annotations_reports_missing_coverage(tmp_path):
    pd.DataFrame(columns=['patient_id','bbox_world']).to_csv(tmp_path/'clinical_tumors.csv',index=False)
    pd.DataFrame([{'case_id':'external-1','crop_mode':'lung_bbox','fallback_reason':''}]).to_csv(tmp_path/'crops.csv',index=False)
    rows,summary=recall_chain.clinical_chain(NS(clinical_meta=str(tmp_path)),{'external-1':{}},.05,5)
    assert rows.empty and list(rows.columns)==['patient_id','crop_mode','fallback',*recall_chain.STAGES]
    row=summary.iloc[0]
    assert (row['n'],row.n_total,row.n_annotated,row.n_unannotated)==(0,1,0,1)
    assert all(pd.isna(row[k]) for k in [*recall_chain.STAGES,'fallback_ratio','truncated_ratio'])


def test_crop_audit_accepts_empty_external_targets(tmp_path,monkeypatch):
    det,clin,proc=tmp_path/'det',tmp_path/'clin',tmp_path/'proc'
    for path in [det,clin,proc]:path.mkdir()
    pd.DataFrame(columns=['seriesuid','split']).to_csv(det/'splits_det.csv',index=False)
    pd.DataFrame(columns=['case_id','crop_mode','fallback_reason']).to_csv(det/'crops.csv',index=False)
    pd.DataFrame(columns=['seriesuid','coordX','coordY','coordZ','diameter_mm']).to_csv(det/'annotations.csv',index=False)
    pd.DataFrame([{'patient_id':'external-1','split':'external'}]).to_csv(clin/'splits_clin.csv',index=False)
    pd.DataFrame(columns=['patient_id','bbox_world']).to_csv(clin/'clinical_tumors.csv',index=False)
    pd.DataFrame([{'case_id':'external-1','crop_mode':'lung_bbox','fallback_reason':''}]).to_csv(clin/'crops.csv',index=False)
    monkeypatch.setattr(crop_rules,'read_header',lambda _:sitk.Image([10,10,10],sitk.sitkInt16))
    cfg=OmegaConf.create({'paths':{'meta':str(det),'processed':str(proc),'luna16_annotations':str(det/'annotations.csv'),'clinical_meta':str(clin),'clinical_processed':str(proc)}})
    crop_rules.audit(cfg)
    out=pd.read_csv(det/'crop_audit.csv')
    assert out.case_id.tolist()==['external-1'] and out.n_targets.tolist()==[0] and out.n_retained.tolist()==[0]
