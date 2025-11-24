# models/unet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import (
    Conv, ResnetBlock, AttentionBlock,
    Downsample, Upsample, SinusoidalPosEmb
)
   
class UNetModel(nn.Module):
    def __init__(self, in_ch=4, base_ch=64, ch_mult=(1,2,4), t_dim=256):
        super().__init__()

        # 1. Step t embedding
        self.t_emb = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim*4),
            nn.SiLU(),
            nn.Linear(t_dim*4, t_dim)
        )

        # 2. Inital conv
        self.conv_in = Conv(in_ch, base_ch, k=3, s=1, p=1)

        # 3. Downsampling layers

        # 4. Middle

        # 5. Upsampling layers

        # 6. Final conv

    def foward(self, x, t):
        # 1. Step t embedding

        # 2. Down path with skip connections

        # 3. Mid

        # 4. Up path
        pass