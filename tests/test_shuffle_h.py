"""Validation-only template/shuffle checks use small CPU tensors and fake IO."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import eval.shuffle_h as shuffle


def target(valid=True):
    return torch.ones(2,2,2), torch.ones(2,2,2), torch.full((2,2,2),float(valid))


def test_invalid_padding_does_not_dilute_a_valid_template():
    good=target()
    bad=(torch.zeros(2,2,2),torch.zeros(2,2,2),torch.zeros(2,2,2))
    pred=shuffle.template_prediction([good,bad])
    assert torch.equal(pred[0],torch.full((2,2,2),.5))
    assert torch.equal(pred[1],torch.ones(2,2,2))
    assert shuffle.lesion_metrics(pred,*good,0)==(1.,0.)


def test_template_weights_each_voxel_and_leaves_unobserved_voxels_zero():
    first=(torch.full((1,1,4),2.),torch.ones(1,1,4),torch.tensor([[[1.,1.,0.,0.]]]))
    second=(torch.full((1,1,4),6.),torch.zeros(1,1,4),torch.tensor([[[0.,1.,1.,0.]]]))
    pred=shuffle.template_prediction([first,second])
    assert torch.equal(pred[1],torch.tensor([[[2.,4.,6.,0.]]]))
    assert torch.equal(pred[0],torch.tensor([[[.5,0.,-.5,-.5]]]))
    assert (pred[0]>0).tolist()==[[[True,False,False,False]]]


def test_unpadded_template_matches_the_original_plain_mean():
    generator=torch.Generator().manual_seed(123)
    targets=[(torch.rand(2,2,2,generator=generator),(torch.rand(2,2,2,generator=generator)>.3).float(),torch.ones(2,2,2)) for _ in range(3)]
    expected=torch.stack([torch.stack([t[1] for t in targets]).mean(0)-.5,
                          torch.stack([t[0] for t in targets]).mean(0)])
    assert torch.allclose(shuffle.template_prediction(targets),expected)


def test_exact_half_mask_occupancy_remains_background():
    first=target()
    second=(torch.ones(2,2,2),torch.zeros(2,2,2),torch.ones(2,2,2))
    pred=shuffle.template_prediction([first,second])
    assert (pred[0]==0).all() and not (pred[0]>0).any()


@pytest.mark.parametrize('targets',[[],[target(False),target(False)]])
def test_template_without_valid_observations_fails_clearly(targets):
    with pytest.raises(ValueError,match='valid recovery'):
        shuffle.template_prediction(targets)


def test_valid_pairs_are_selected_before_stacking_and_shuffling():
    hs=[torch.tensor([1.,2.]),torch.tensor([999.,999.]),torch.tensor([3.,4.])]
    targets=[target(),target(False),target()]
    h,kept=shuffle.comparison_inputs(hs,targets)
    assert torch.equal(h,torch.tensor([[1.,2.],[3.,4.]]))
    assert torch.equal(h.roll(1,0),torch.tensor([[3.,4.],[1.,2.]]))
    assert len(kept)==2 and kept[0] is targets[0] and kept[1] is targets[2]


@pytest.mark.parametrize('flags',[[],[True],[True,False],[False,False]])
def test_nontrivial_shuffle_requires_two_valid_targets(flags):
    with pytest.raises(ValueError,match='at least two valid recovery targets'):
        shuffle.comparison_inputs([torch.zeros(2) for _ in flags],[target(flag) for flag in flags])


def patch_main(monkeypatch,hs,targets,seen):
    cfg=SimpleNamespace(model=SimpleNamespace(dim=1,recovery=SimpleNamespace(dilate_vox=0)),
                        patch_zyx=[4,4,4],train=SimpleNamespace(batch=1,amp=False))
    monkeypatch.setattr(shuffle,'load_detector',lambda *args:(cfg,lambda *args:{}))
    monkeypatch.setattr(shuffle.torch,'load',lambda *args,**kwargs:{'recovery':{}})
    original_tensor=torch.tensor

    def cpu_tensor(*args,**kwargs):
        if kwargs.get('device')=='cuda':kwargs['device']='cpu'
        return original_tensor(*args,**kwargs)

    monkeypatch.setattr(shuffle.torch,'tensor',cpu_tensor)
    monkeypatch.setattr(shuffle.torch,'autocast',lambda *args,**kwargs:nullcontext())

    class Recovery:
        def __init__(self,dim):pass
        def to(self,device):return self
        def eval(self):return self
        def load_state_dict(self,state):pass
        def __call__(self,h):
            seen.setdefault('features',[]).append(h.clone())
            return torch.stack([torch.full((len(h),2,2,2),.5),torch.ones(len(h),2,2,2)],1)

    case=SimpleNamespace(centers=np.zeros((1,3)),has_mask=np.array([True]))
    ds=SimpleNamespace(ids=['validation-case'],case=lambda index:case,patch_at=lambda *args:None)
    monkeypatch.setattr(shuffle,'RecoveryDecoder',Recovery)
    monkeypatch.setattr(shuffle,'luna_dataset',lambda cfg,split,augment:seen.update(split=split,augment=augment) or ds)
    monkeypatch.setattr(shuffle,'collate',lambda samples:{'image':None,'tokens':None,'token_xyz':None,'boxes':[]})
    monkeypatch.setattr(shuffle,'to_device',lambda batch,device:batch)
    monkeypatch.setattr(shuffle,'match_layers',lambda *args:[[]])
    monkeypatch.setattr(shuffle,'recovery_inputs',lambda *args:(hs,targets))
    monkeypatch.setattr(shuffle,'write_json',lambda path,payload:seen.update(output=path,payload=payload))


def test_main_shuffles_only_valid_features_and_uses_validation(monkeypatch):
    hs=[torch.tensor([1.]),torch.tensor([999.]),torch.tensor([2.])]
    targets=[target(),target(False),target()];seen={}
    patch_main(monkeypatch,hs,targets,seen)
    shuffle.main(Path('/synthetic/run'))
    assert seen['split']=='val' and seen['augment'] is False
    assert torch.equal(seen['features'][0],torch.tensor([[1.],[2.]]))
    assert torch.equal(seen['features'][1],torch.tensor([[2.],[1.]]))
    assert seen['payload']['n_lesions']==2
    assert set(seen['payload'])=={'n_lesions','true','shuffled','template'}


def test_main_with_no_usable_targets_does_not_write_a_fake_comparison(monkeypatch):
    seen={};patch_main(monkeypatch,[],[],seen)
    with pytest.raises(ValueError,match='at least two valid recovery targets; found 0'):
        shuffle.main(Path('/synthetic/run'))
    assert 'payload' not in seen and 'features' not in seen
