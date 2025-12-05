# models/blocks.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Normalize(nn.Module):
    def __init__(self, ch, groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(groups, ch)

    def forward(self, x):
        return self.norm(x)

class ResnetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0, t_dim=None):
        super().__init__()

        self.t_mlp = nn.Linear(t_dim, out_ch) if t_dim else None

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

    def forward(self, x, t_emb=None):
        h = self.conv1(self.act1(self.norm1(x)))

        # broadcasting (B,C,H,W) + (B,C,1,1)
        h = h + (self.t_mlp(t_emb)[:, :, None, None] if t_emb is not None else 0)

        h = self.dropout(h)
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)

class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1): # kernel size, stride, padding
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = Conv(ch, ch, k=3, s=1, p=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest") #CHW -> C H*2 W*2
        return self.conv(x)
    
class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()

        #CHW -> C H/2 W/2
        self.conv = Conv(ch, ch, k=3, s=2, p=1) # stride = 2

    def forward(self, x):
        return self.conv(x)
    
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: B
        half = self.dim // 2 # half dim for sin, haft dim for cos
        
        # torch.arange -> [0, 1, 2, 3,..., half-1] 
        # c = (-ln(100000) / (half - 1))
        # emb = exp([0, 1, 2, 3,..., half] * c)
        # emb = [0, -3.0701, -6.1402, -9.2103]
        emb = torch.exp(
            torch.arange(half, dtype=torch.float32, device=t.device) * 
            -(torch.log(torch.tensor(10000.0)) / (half - 1))
        )

        # broadcasts 
        # emb: (half,) -> emb[None, :] -> (half,1)
        # t: (B,) -> t[:, None] -> (B,1)
        # emb: (B,1) * (1,half) -> (B,half)
        # emb[i][j] = t[i] * freq[j]
        emb = t[:, None] * emb[None, :]

        # emb: (B,2*half) -> (B,dim)
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        return emb


class AttentionBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = Normalize(ch)
        self.q = Conv(ch, ch, k=1, s=1, p=0)
        self.k = Conv(ch, ch, k=1, s=1, p=0)
        self.v = Conv(ch, ch, k=1, s=1, p=0)
        self.proj = Conv(ch, ch, k=1, s=1, p=0)

    def forward(self, x):
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)

        b, c, h_, w_ = q.shape
        q = q.reshape(b, c, h_*w_).transpose(1,2)   # (B,C,H,W) -> (B,HW,C)
        k = k.reshape(b, c, h_*w_)                  # (B,C,HW)
        v = v.reshape(b, c, h_*w_).transpose(1,2)   # (B,C,H,W) -> (B,HW,C)

        qk = (q @ k) / (c ** 0.5) # (B,HW,HW) qk/sqrt(dim)
        attn = torch.softmax(qk, dim=-1) # (B,HW,HW)
        
        out = attn @ v # (B,HW,HW) * (B,HW,C) -> (B,HW,C)
        
        # (B,HW,C) -> (B,C,HW) -> (B,C,H,W)
        out = out.transpose(1,2).reshape(b, c, h_, w_)
        return x + self.proj(out)
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, ch, num_head=8):
        super().__init__()
        assert ch % num_head == 0, "channels must be divisible by num_head"

        self.norm =nn.GroupNorm(32, ch)
        self.num_head = num_head
        self.head_dim = ch // num_head
        self.scale = self.head_dim ** -0.5

        self.to_q = Conv(ch, ch, k=1, s=1, p=0)
        self.to_k = Conv(ch, ch, k=1, s=1, p=0)
        self.to_v = Conv(ch, ch, k=1, s=1, p=0)
        self.to_out = Conv(ch, ch, k=1, s=1, p=0)

    def forward(self, x):
        b, c, h, w, = x.shape

        x_norm = self.norm(x)

        q = self.to_q(x_norm)
        k = self.to_k(x_norm)
        v = self.to_v(x_norm)

        # (B, C, H, W) -> (B, num_head, HW, head_dim)
        q = q.view(b, self.num_head, self.head_dim, h*w).transpose(2, 3)
        k = k.view(b, self.num_head, self.head_dim, h*w)
        v = v.view(b, self.num_head, self.head_dim, h*w).transpose(2, 3)

        attn = (q @ k) * self.scale # (B, num_head, HW, HW)
        attn = torch.softmax(attn, dim=-1)

        out = attn @ v
        out = out.transpose(2, 3).contiuous().view(b,c,h,w)

        return x + self.to_out(out)
