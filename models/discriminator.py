# models/disciminator.py

import math
import torch
import torch.nn as nn
from models.blocks import Conv

class Discriminator(nn.Module):
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()

        self.model = nn.Sequential(
            # 64x64 -> 32x32
            Conv(in_ch, base_ch, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            Conv(base_ch, base_ch*2, 4, 2, 1),
            nn.BatchNorm2d(base_ch*2),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            Conv(base_ch*2, base_ch*4, 4, 2, 1),
            nn.BatchNorm2d(base_ch*4),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            Conv(base_ch*4, base_ch*8, 4, 2, 1),
            nn.BatchNorm2d(base_ch*8),
            nn.LeakyReLU(0.2, inplace=True),

            # final output: (B,1,4,4) logits
            Conv(base_ch*8, 1, 3, 1, 1)
        )

    def forward(self, x):
        return self.model(x)
