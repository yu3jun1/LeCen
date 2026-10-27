"""3D ResNet-18 风格编码器 + FPN。输入 (B, 1, 80, 192, 192)，输出 {2: P2, 4: P4, 8: P8}，通道 D，步长按体素计。"""
import torch.nn as nn
import torch.nn.functional as F


def norm(c):
    return nn.GroupNorm(16, c)


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = norm(cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = norm(cout)
        self.down = (nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False), norm(cout))
                     if stride != 1 or cin != cout else nn.Identity())

    def forward(self, x):
        y = F.relu(self.n1(self.conv1(x)))
        return F.relu(self.n2(self.conv2(y)) + self.down(x))


class ResNet18FPN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, 32, 3, 2, 1, bias=False), norm(32), nn.ReLU(inplace=True))
        chans, strides = (64, 128, 256, 512), (1, 2, 2, 2)  # C2, C4, C8, C16
        self.stages = nn.ModuleList()
        cin = 32
        for c, s in zip(chans, strides):
            self.stages.append(nn.Sequential(BasicBlock(cin, c, s), BasicBlock(c, c, 1)))
            cin = c
        self.lateral = nn.ModuleList(nn.Conv3d(c, dim, 1) for c in chans)
        self.smooth = nn.ModuleList(nn.Conv3d(dim, dim, 3, 1, 1) for _ in range(3))

    def forward(self, x):
        feats, y = [], self.stem(x)
        for stage in self.stages:
            y = stage(y)
            feats.append(y)
        p = self.lateral[3](feats[3])
        out = {}
        for i in (2, 1, 0):
            p = self.lateral[i](feats[i]) + F.interpolate(p, size=feats[i].shape[2:], mode="nearest")
            out[2 ** (i + 1)] = self.smooth[i](p)
        return out
