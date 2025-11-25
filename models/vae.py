# models/vae.py

import torch
import torch.nn as nn
from models.blocks import Conv, ResnetBlock, Downsample, Upsample

class Encoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, z_ch=4):
        super().__init__()

        # 3 x 64 x 64
        self.conv_in = Conv(in_ch, base_ch) # 64 x 64 x 64

        self.blocks = nn.ModuleList([
            ResnetBlock(base_ch, base_ch), # 64 x 64 x 64
            Downsample(base_ch), # (H/2 W/2) --> 64 x 32 x 32

            ResnetBlock(base_ch, base_ch*2), # 128 x 32 x 32
            Downsample(base_ch*2),# H/4 W/4 # 128 x 16 x 16

            ResnetBlock(base_ch*2, base_ch*4), # 256 x 16 x 16
        ])

        # 4C H/4 W/4 (4 x 16 x 16)
        # 2z: mean and log(var) -> sampling latent  
        self.conv_out = Conv(base_ch*4, 2*z_ch, k=3, s=1, p=1) 

    def forward(self, x):
        h = self.conv_in(x)
        for blk in self.blocks:
            h = blk(h)
        return self.conv_out(h)

class Decoder(nn.Module):
    def __init__(self, out_ch=3, base_ch=64, z_ch=4):
        super().__init__()
        self.conv_in = Conv(z_ch, base_ch*4)

        self.blocks = nn.ModuleList([
            ResnetBlock(base_ch*4, base_ch*4),
            Upsample(base_ch*4), # 256 2h 2w

            ResnetBlock(base_ch*4, base_ch*2),
            Upsample(base_ch*2), # 128 4h 4w = H W

            ResnetBlock(base_ch*2, base_ch),
        ])

        # original shape (3,H,W)
        self.conv_out = Conv(base_ch, out_ch, k=3, s=1, p=1)

    def forward(self, x):
        h = self.conv_in(x)
        for blk in self.blocks:
            h = blk(h)
        return self.conv_out(h)

class AutoEncoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, z_ch=4):
        super().__init__()
        self.encoder = Encoder(in_ch, base_ch, z_ch)
        self.decoder = Decoder(in_ch, base_ch, z_ch)

    def encode(self, x):
        h = self.encoder(x) # C H/4 W/4
        mu, logvar = torch.chunk(h, 2, dim=1)
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
        h = self.encode(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar