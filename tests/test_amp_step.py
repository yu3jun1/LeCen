"""裁剪前已 unscale；inf 步不推进 scheduler 与步数。"""
import torch

from engine.train_det import optimizer_step


def setup():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    scaler = torch.amp.GradScaler("cpu", init_scale=1024.0)
    return model, opt, sched, scaler


def test_clip_after_unscale():
    model, opt, sched, scaler = setup()
    x = torch.randn(8, 4)
    model(x).sum().backward()
    true_norm = torch.cat([p.grad.flatten() for p in model.parameters()]).norm()
    opt.zero_grad()
    scaler.scale(model(x).sum()).backward()
    stepped, norm = optimizer_step(list(model.parameters()), opt, scaler, sched, clip_norm=1.0)
    assert stepped and abs(norm - float(true_norm)) < 1e-4
    assert sched.last_epoch == 1


def test_inf_step_skipped():
    model, opt, sched, scaler = setup()
    before = [p.detach().clone() for p in model.parameters()]
    scaler.scale(model(torch.randn(8, 4)).sum() * float("inf")).backward()
    stepped, _ = optimizer_step(list(model.parameters()), opt, scaler, sched, clip_norm=1.0)
    assert not stepped and sched.last_epoch == 0
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
