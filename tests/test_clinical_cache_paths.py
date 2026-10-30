"""Clinical entry points use current caches without rewriting completed detector runs."""
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import torch

from common import io
from engine import adapt_clinical, extract_slots, infer_det
from eval import recall_chain


def current_paths(tmp_path):
    return OmegaConf.create({'clinical_meta':str(tmp_path/'current-meta'),
                             'clinical_processed':str(tmp_path/'current-processed'),
                             'cache':str(tmp_path/'current-cache')})


def test_slot_tokens_hashes_and_output_use_current_cache(tmp_path,monkeypatch):
    paths=current_paths(tmp_path)
    cfg=OmegaConf.create({'paths':{'meta':'original-luna-meta','processed':'original-luna-processed',
                                  'cache':str(tmp_path/'old-cache')},
                          'preprocess':{'version':'synthetic'},'patch_zyx':[16,16,16],
                          'infer':{'overlap':.25}})
    original=OmegaConf.to_container(cfg,resolve=True)
    tables={'splits_clin.csv':pd.DataFrame([{'patient_id':'L1','split':'val'}]),
            'clinical.csv':pd.DataFrame([{'patient_id':'L1','series_uid':'series-L1'}])}
    monkeypatch.setattr(extract_slots.pd,'read_csv',lambda path,**kwargs:tables[Path(path).name])
    monkeypatch.setattr(extract_slots,'load_detector',lambda *args:(cfg,None))
    monkeypatch.setattr(extract_slots,'read_json',lambda path:{'p_min':.05})
    monkeypatch.setattr(extract_slots,'extractor_hash',lambda *args:'extractor')
    hashed=[]
    monkeypatch.setattr(extract_slots,'file_sha1',lambda path:hashed.append(Path(path)) or 'sha1')
    seen={}
    monkeypatch.setattr(extract_slots,'InferCase',lambda patient,vol_dir,token_dir,*args:seen.update(vol_dir=vol_dir,token_dir=token_dir))
    monkeypatch.setattr(extract_slots,'infer_case',lambda *args:{'keep':np.array([0]),'dec_p':np.array([.5],np.float32),
                          'dec_h':np.zeros((1,4),np.float32),'dec_box':np.zeros((1,6),np.float32)})
    monkeypatch.setattr(extract_slots,'write_npz',lambda path,**arrays:seen.update(output=Path(path)))
    _,out=extract_slots.extract(tmp_path/'completed-detector',paths,'cpu')
    assert seen['token_dir']==Path(paths.cache)/'ctclip'
    assert seen['vol_dir']==Path(paths.clinical_processed)/'clinical'
    assert Path(paths.cache)/'ctclip/L1.npz' in hashed
    assert out==Path(paths.cache)/'slots/extractor'
    assert seen['output']==out/'L1.npz'
    assert OmegaConf.to_container(cfg,resolve=True)==original


def test_adaptation_records_current_cache_before_make_run_and_keeps_luna_paths(tmp_path,monkeypatch):
    src=tmp_path/'completed-detector';src.mkdir()
    cfg=OmegaConf.create({'name':'D1','train':{'iters':1,'amp':False},'adapt':{'iters':0},
                         'model':{'recovery':{'enabled':False}},'seeds':{'data':2000},
                         'patch_zyx':[16,16,16], 'paths':{'meta':'pinned-luna-meta','processed':'pinned-luna-processed',
                         'cache':'old-cache'}})
    OmegaConf.save(cfg,src/'config.yaml');(src/'p_min.json').write_text('{"p_min":0.05}\n')
    original=(src/'config.yaml').read_bytes()
    paths=current_paths(tmp_path);Path(paths.clinical_meta).mkdir()
    pd.DataFrame([{'patient_id':'L1','split':'train'},{'patient_id':'A1','split':'external'}]).to_csv(Path(paths.clinical_meta)/'splits_clin.csv',index=False)
    monkeypatch.setattr(adapt_clinical,'load_config',lambda path:OmegaConf.create({'paths':paths}))
    monkeypatch.setattr(adapt_clinical,'set_determinism',lambda:None)
    monkeypatch.setattr(adapt_clinical,'setup',lambda:torch.device('cpu'))
    monkeypatch.setattr(adapt_clinical,'world',lambda:1)
    monkeypatch.setattr(adapt_clinical,'is_main',lambda:True)
    monkeypatch.setattr(adapt_clinical.torch,'load',lambda *args,**kwargs:{'model':{}})
    monkeypatch.setattr(adapt_clinical,'file_sha1',lambda path:'source-sha1')
    observed={};out=tmp_path/'new-adaptation'

    def make_run(effective,hashes):
        observed['saved_paths']=OmegaConf.to_container(effective.paths,resolve=True)
        out.mkdir()
        return out

    class Model:
        def load_state_dict(self,state):pass
        def to(self,device):assert device.type=='cpu'

    class Step:
        def __init__(self,*args):pass
        def train(self):return self
        def parameters(self):return []

    monkeypatch.setattr(adapt_clinical,'make_run',make_run)
    monkeypatch.setattr(adapt_clinical,'Detector',lambda cfg:Model())
    monkeypatch.setattr(adapt_clinical,'TrainStep',Step)
    monkeypatch.setattr(adapt_clinical,'wrap',lambda model,device:model)
    monkeypatch.setattr(adapt_clinical,'build_optim',lambda *args:(None,None))
    monkeypatch.setattr(adapt_clinical,'loader',lambda *args:iter(()))
    monkeypatch.setattr(adapt_clinical,'save',lambda *args:None)

    def clinical_dataset(effective,ids,volume,label,targets,augment):
        observed.update(clinical_ids=list(ids),clinical_cache=effective.paths.cache,clinical_volume=volume)
        return None

    def luna_dataset(effective,*args,**kwargs):
        observed.update(luna_meta=effective.paths.meta,luna_processed=effective.paths.processed,luna_cache=effective.paths.cache)
        return None

    monkeypatch.setattr(adapt_clinical,'DetPatchDataset',clinical_dataset)
    monkeypatch.setattr(adapt_clinical,'luna_dataset',luna_dataset)
    assert adapt_clinical.main(src,['adapt.iters=0'])==out
    assert observed['saved_paths']['cache']==paths.cache
    assert observed['saved_paths']['clinical_meta']==paths.clinical_meta
    assert observed['saved_paths']['clinical_processed']==paths.clinical_processed
    assert observed['clinical_cache']==paths.cache and observed['luna_cache']==paths.cache
    assert observed['luna_meta']=='pinned-luna-meta' and observed['luna_processed']=='pinned-luna-processed'
    assert observed['clinical_ids']==['L1']
    assert (src/'config.yaml').read_bytes()==original


def test_clinical_recall_cli_predict_reads_current_cache_and_preserves_empty_report(tmp_path,monkeypatch):
    paths=current_paths(tmp_path);Path(paths.clinical_meta).mkdir()
    pd.DataFrame([{'patient_id':'A1','split':'external'}]).to_csv(Path(paths.clinical_meta)/'splits_clin.csv',index=False)
    pd.DataFrame(columns=['patient_id','bbox_world']).to_csv(Path(paths.clinical_meta)/'clinical_tumors.csv',index=False)
    pd.DataFrame([{'case_id':'A1','crop_mode':'body','fallback_reason':''}]).to_csv(Path(paths.clinical_meta)/'crops.csv',index=False)
    cfg=OmegaConf.create({'paths':{'cache':'old-cache','meta':'pinned-luna-meta','processed':'pinned-luna-processed'}})
    src=tmp_path/'completed-detector';src.mkdir();OmegaConf.save(cfg,src/'config.yaml')
    original=(src/'config.yaml').read_bytes()
    monkeypatch.setattr(infer_det,'load_detector',lambda *args:(cfg,None))
    observed={}

    def predict(model,effective,ids,volume,device):
        observed.update(cache=effective.paths.cache,meta=effective.paths.meta,processed=effective.paths.processed,
                        ids=list(ids),volume=volume)
        return {'A1':{}}

    monkeypatch.setattr(infer_det,'predict',predict)
    monkeypatch.setattr(io,'load_config',lambda path:OmegaConf.create({'paths':paths,'slots':{'cover_margin_mm':5}}))
    monkeypatch.setattr(io,'read_json',lambda path:{'p_min':.05})
    written={}
    monkeypatch.setattr(io,'write_csv',lambda path,frame:written.update({Path(path).name:frame.copy()}))
    monkeypatch.setattr(sys,'argv',['eval.recall_chain','--run',str(src),'--split','external','--cohort','clinical'])
    runpy.run_path(recall_chain.__file__,run_name='__main__')
    assert observed['cache']==paths.cache and observed['volume']==Path(paths.clinical_processed)/'clinical'
    assert observed['meta']=='pinned-luna-meta' and observed['processed']=='pinned-luna-processed'
    assert observed['ids']==['A1']
    summary=written['recall_chain_clinical_external.csv'].iloc[0]
    assert summary['n_annotated']==0 and summary['n_unannotated']==1
    assert summary['work'] is None
    assert (src/'config.yaml').read_bytes()==original
