# models/vae.py

import math
import torch
import torch.nn as nn
from models.blocks import Conv, ResnetBlock, Downsample, Upsample

class Encoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, z_ch=4, down_factor=4):
        super().__init__()

        assert down_factor in [2,4,8], "down_factor must be 2,4,8"
        num_down = int(math.log2(down_factor))

        self.conv_in = Conv(in_ch, base_ch) 

        ch = base_ch
        self.blocks = nn.ModuleList()
        
        # First base _ch resnet
        self.blocks.append(ResnetBlock(ch, ch))
        self.blocks.append(Downsample(ch))

        for _ in range(num_down-1):
            self.blocks.append(ResnetBlock(ch, ch*2))
            self.blocks.append(Downsample(ch*2))
            ch *= 2

        self.blocks.append(ResnetBlock(ch, ch * 2))

        # (2z, H/factor, W/factor)
        # 2z: mean and log(var) -> sampling latent
        final_ch = base_ch * down_factor
        self.conv_out = Conv(final_ch, 2*z_ch, k=3, s=1, p=1) 

    def forward(self, x):
        h = self.conv_in(x)
        for blk in self.blocks:
            h = blk(h)
        return self.conv_out(h)

class Decoder(nn.Module):
    def __init__(self, out_ch=3, base_ch=64, z_ch=4, up_factor=4):
        super().__init__()

        assert up_factor in [2,4,8], "up_factor must be 2,4,8"
        num_up = int(math.log2(up_factor))

        ch = base_ch * up_factor
        self.conv_in = Conv(z_ch, ch)

        self.blocks = nn.ModuleList()
        
        self.blocks.append(ResnetBlock(ch, ch//2))
        
        for _ in range(num_up-1):
            ch //= 2
            self.blocks.append(Upsample(ch))
            self.blocks.append(ResnetBlock(ch, ch//2))

        ch //= 2
        self.blocks.append(Upsample(ch))
        self.blocks.append(ResnetBlock(ch, ch))

        # original shape (3,H,W)
        self.conv_out = Conv(base_ch, out_ch, k=3, s=1, p=1)

    def forward(self, x):
        h = self.conv_in(x)
        for blk in self.blocks:
            h = blk(h)
        return self.conv_out(h)

class AutoEncoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, z_ch=4, factor=4):
        super().__init__()
        self.encoder = Encoder(in_ch, base_ch, z_ch, factor)
        self.decoder = Decoder(in_ch, base_ch, z_ch, factor)

    def encode(self, x):
        h = self.encoder(x)  # (B, 2*z_ch, H', W')
        mu, logvar = torch.chunk(h, 2, dim=1)  # each (B, z_ch, H', W')
        return mu, logvar
    
    def encode_to_z(self, x):
        """Return the sampled latent z from input x."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        # eps ~ N(0,1), std arg just for the shape not value
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)          # get mu, logvar
        z = self.reparameterize(mu, logvar)  # sample once
        recon = self.decode(z)
        return recon, mu, logvar