from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional


@dataclass
class ShiftConfig:
    seqLen: int
    shift: Optional[int] = None

class Shift(nn.Module):
    """
    Use Linear\Relu layers with a final sigmoid activation to
    determine how much to 'shift' or 'roll' the time sequence
    """

    def __init__(self, configs:ShiftConfig):
        super(Shift, self).__init__()
        self.configs = configs
        self.shiftPred = nn.Sequential(
            nn.Linear(configs.seqLen, configs.seqLen // 2, bias=True),
            nn.ReLu(),
            nn.Linear(configs.seqLen // 2, configs.seqLen // 4, bias=True),
            nn.ReLu(),
            nn.Linear(configs.seqLen // 4, configs.seqLen // 8, bias=True),
            nn.ReLu(),
            nn.Linear(configs.seqLen // 8, 1, bias=True),
            F.sigmoid()
        )

    def forward(self, x):

        _, L, _ = x.shape # B L N
        # B: batch_size;
        # L: seq_len;
        # N: number of variate (tokens), can also includes covariates
        assert L == self.configs.seqLen

        # Shift determination
        # B L N -> B N L
        x_shift = x.permute(0, 2, 1)

        # B L N -> B N 1
        x_shift = self.shiftPred(x_shift)
        y_shift = x_shift * self.configs.seqLen
        y_shift = torch.round(y_shift).long()

        y = torch.roll(x, y_shift, dims=1)

        return y  # [B, L(shifted by y_shift), N]
