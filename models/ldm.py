# models/ldm.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ddpm import DDPMScheduler
from models.unet import UNet
from models.vae import AutoEncoder


class LatentDiffusion(nn.Module):
    def __init__(
        self, 
        unet: UNet, 
        vae: AutoEncoder, 
        scheduler: DDPMScheduler
    ):
        super().__init__()
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler

    def encode(self, x):
        z = self.vae.encode(x)
        return z
    
    def decode(self, z):
        x = self.vae.decode(z)
        return x
    
    def forward(self, images):
        z_0 = self.encode(images)

        B = z_0.size(0)
        device = z_0.device

        t = torch.randint(
            low=0,
            high=self.scheduler.num_steps,
            size=(B,),
            device=device,
        )

        gaus_noise = torch.randn_like(z_0)

        z_t = self.scheduler.add_noise(z_0, gaus_noise, t)

        pred_noise = self.unet(z_t, t)

        loss = F.mse_loss(pred_noise, gaus_noise)

        return loss, pred_noise, gaus_noise, t
    
    @torch.no_grad()
    def sample(self, num_samples, latent_shape=(4, 64, 64), device=None):
        if device is None:
            device = next(self.parameters()).device

        C, H, W = latent_shape

        z_t = torch.randn(num_samples, C, H, W, device=device)

        for t in reversed(range(self.scheduler.num_steps)):
            t_tensor = torch.full(
                (num_samples,),
                t,
                device=device,
                dtype=torch.long
            )

            noise_pred = self.unet(z_t, t_tensor)

            z_t = self.scheduler.step(z_t, noise_pred, t_tensor)
        
        x_rec = self.decode(z_t)
        return x_rec