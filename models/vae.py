import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv

ptdtype = {None: torch.float32, 'fp32': torch.float32, 'bf16': torch.bfloat16}

class Normalize(nn.Module):
    def __init__(self, ch, groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(groups, ch)

    def forward(self, x):
        return self.norm(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()

        self.norm1 = Normalize(in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = Conv(in_ch, out_ch, k=3, s=1, p=1)

        self.norm2 = Normalize(out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = Conv(out_ch, out_ch, k=3, s=1, p=1)

        if in_ch != out_ch:
            self.skip = Conv(in_ch, out_ch, k=1, s=1, p=0)
        else:
            self.skip = nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.dropout(h)
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)

class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()

        #CHW -> C H/2 W/2
        self.conv = Conv(ch, ch, k=3, s=2, p=1) # stride = 2

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = Conv(ch, ch, k=3, s=1, p=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest") #CHW -> C H*2 W*2
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, z_ch=4):
        super().__init__()

        self.conv_in = Conv(in_ch, base_ch)

        self.blocks = nn.ModuleList([
            ResnetBlock(base_ch, base_ch),
            Downsample(base_ch), # H/2 W/2

            ResnetBlock(base_ch, base_ch*2),
            Downsample(base_ch*2),# H/4 W/4

            ResnetBlock(base_ch*2, base_ch*4),
        ])

        # 4C H/4 W/4, 2z: mean and log(var) -> sampling latent  
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
            Upsample(base_ch*4), # 2H 2W

            ResnetBlock(base_ch*4, base_ch*2),
            Upsample(base_ch*2), # 4H 4W

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
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        # eps ~ N(0,1), std arg just for the shape not value
        eps = torch.randn_like(std) 
        return mu + eps*std

    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar