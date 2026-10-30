"""A failure snapshot must preserve the state needed to reproduce a failed forward."""
from omegaconf import OmegaConf
import torch

from engine.train_det import save_failure


def test_failure_snapshot_roundtrip(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10)
    batch = {'image': torch.tensor([[2., 3.]]), 'seriesuid': ['fixture']}
    cfg = OmegaConf.create({'name': 'D1', 'train': {'failure_dir': str(tmp_path)}})
    save_failure(cfg, model, None, optimizer, scheduler, batch, 12, ValueError('nonfinite cost'))
    state = torch.load(next(tmp_path.glob('*.pt')), weights_only=True, map_location='cpu')
    restored = torch.nn.Linear(2, 1)
    restored.load_state_dict(state['model'])
    torch.testing.assert_close(restored(batch['image']), model(batch['image']))
    assert state['optimizer']['state'] and state['step'] == 12
    torch.testing.assert_close(state['batch']['image'], batch['image'])
    assert state['error'] == 'nonfinite cost'
    assert state['gradients']
    assert 'scaler' not in state and 'skipped' not in state


def test_failure_snapshot_is_opt_in(tmp_path):
    cfg = OmegaConf.create({'name': 'D1', 'train': {}})
    save_failure(cfg, None, None, None, None, None, 0, ValueError('unused'))
    assert not list(tmp_path.iterdir())
