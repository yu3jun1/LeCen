"""恢复 decoder：只接受 h，不连接 backbone 特征。输出 (M, 2, 32, 32, 32)：mask logit 与归一化强度。"""
import torch.nn as nn


class RecoveryDecoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 128 * 4 ** 3)
        chans = (128, 64, 32, 16)
        self.up = nn.Sequential(*[nn.Sequential(nn.ConvTranspose3d(a, b, 4, 2, 1), nn.GroupNorm(8, b), nn.GELU())
                                  for a, b in zip(chans, chans[1:])])
        self.out = nn.Conv3d(chans[-1], 2, 1)

    def forward(self, h):
        return self.out(self.up(self.fc(h).view(-1, 128, 4, 4, 4)))
