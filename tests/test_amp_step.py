"""BF16 uses ordinary backward; invalid gradients must fail before any update."""
import copy
import pytest
import torch
from engine.train_det import optimizer_step


def test_clip_before_update():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    model(torch.randn(8, 4)).sum().backward()
    gradients = [p.grad.clone() for p in model.parameters()]
    before = [p.detach().clone() for p in model.parameters()]
    expected_norm = torch.cat([g.flatten() for g in gradients]).norm()
    norm = optimizer_step(list(model.parameters()), optimizer, scheduler, 1.0)
    assert abs(norm - float(expected_norm)) < 1e-5
    factor = min(1.0, 1.0 / (float(expected_norm) + 1e-6))
    for old, gradient, parameter in zip(before, gradients, model.parameters()):
        torch.testing.assert_close(parameter, old - 0.1 * gradient * factor)
        assert parameter.grad is None
    assert scheduler.last_epoch == 1


@pytest.mark.parametrize('value', [float('inf'), float('nan')])
def test_nonfinite_gradients_preserve_parameters_optimizer_and_scheduler(value):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10)
    model(torch.ones(1, 2)).sum().backward()
    optimizer_step(list(model.parameters()), optimizer, scheduler, 1.0)
    before = [p.detach().clone() for p in model.parameters()]
    state = copy.deepcopy(optimizer.state_dict())
    epoch = scheduler.last_epoch
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, value)
    with pytest.raises(RuntimeError, match='non-finite'):
        optimizer_step(list(model.parameters()), optimizer, scheduler, 1.0)
    assert scheduler.last_epoch == epoch
    for old, parameter in zip(before, model.parameters()):
        assert torch.equal(old, parameter)
    for key, expected in state['state'].items():
        for name, tensor in expected.items():
            torch.testing.assert_close(optimizer.state_dict()['state'][key][name], tensor)
