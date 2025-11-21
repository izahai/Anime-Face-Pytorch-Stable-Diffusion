import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1): # kernel size, stride, padding
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)

    def forward(self, x):
        return self.conv(x)
