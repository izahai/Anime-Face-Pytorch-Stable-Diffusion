# models/unet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import (
    Conv, ResnetBlock, AttentionBlock,
    Downsample, Upsample, SinusoidalPosEmb
)
   
class UNet(nn.Module):
    def __init__(self, in_ch=4, base_ch=64, ch_mult=(1,2,4), t_dim=256):
        super().__init__()

        self.ch_mult = ch_mult
        self.num_stage = len(ch_mult)
        self.down_channels = [base_ch * mult for mult in ch_mult]

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
        self.down_blocks = nn.ModuleDict()
        ch = base_ch
        for i, mult in enumerate(ch_mult):
            out_ch = base_ch * mult

            self.down_blocks[f"down_{i}_resblock_0"] = ResnetBlock(ch, out_ch, t_dim=t_dim)
            self.down_blocks[f"down_{i}_resblock_1"] = ResnetBlock(out_ch, out_ch, t_dim=t_dim)
            
            # Downsample at every stage EXCEPT the last one.
            if mult != ch_mult[-1]:
                self.down_blocks[f"downsample_{i}"] = Downsample(out_ch)
            ch = out_ch

        # 4. Middle
        ch = self.down_channels[-1]
        self.mid_blocks = nn.ModuleDict({
            "mid_resblock_0": ResnetBlock(ch, ch, t_dim=t_dim),
            "mid_attn": AttentionBlock(ch),
            "mid_resblock_1": ResnetBlock(ch, ch, t_dim=t_dim),
        })

        # 5. Upsampling layers
        self.up_blocks = nn.ModuleDict()

        for i, mult in enumerate(reversed(ch_mult)):
            skip_ch = self.down_channels[-1 - i]
            out_ch = base_ch * mult
            
            # ch: for previous layer
            # skip_ch: for the last layer at correspoding stage in downsample layer 
            # (ch + skip_ch) for concatenation skip connections (ckc)
            # only the "down_{i}_resblock_0" accept ckc
            self.up_blocks[f"up_{i}_resblock_0"] = ResnetBlock(ch + skip_ch, out_ch, t_dim=t_dim)
            self.up_blocks[f"up_{i}_resblock_1"] = ResnetBlock(out_ch, out_ch, t_dim=t_dim)

            # Upsample at every stage EXCEPT the last one.
            if mult != ch_mult[0]:
                self.up_blocks[f"upsample_{i}"] = Upsample(out_ch)
            ch = out_ch

        # 6. Final conv
        self.conv_out = Conv(ch, in_ch, k=3, s=1, p=1)

    def forward(self, x, t):
        # 1. Step t embedding
        t_emb = self.t_emb(t) # (B, t_dim)

        # 2. Down
        skips = []
        h = self.conv_in(x)
        for i in range(self.num_stage):
            h = self.down_blocks[f"down_{i}_resblock_0"](h, t_emb)
            h = self.down_blocks[f"down_{i}_resblock_1"](h, t_emb)
            skips.append(h)
            if i != self.num_stage-1:
                h = self.down_blocks[f"downsample_{i}"](h)
        
        # 3. Mid
        h = self.mid_blocks["mid_resblock_0"](h, t_emb)
        h = self.mid_blocks["mid_attn"](h)
        h = self.mid_blocks["mid_resblock_1"](h, t_emb)

        # 4. Up
        for i in range(self.num_stage):
            h = torch.cat([h, skips.pop()], dim=1)
            h = self.up_blocks[f"up_{i}_resblock_0"](h, t_emb)
            h = self.up_blocks[f"up_{i}_resblock_1"](h, t_emb)
            if i != self.num_stage-1:
                h = self.up_blocks[f"upsample_{i}"](h)
        
        return self.conv_out(h)