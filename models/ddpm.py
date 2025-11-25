# models/ddpm.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DDPMScheduler:
    def __init__(self, 
        num_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device=None,
    ):
        self.num_steps = num_steps
        self.device = device
    
        # Beta schedule (linear)
        betas =  torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
        if device is not None:
            betas = betas.to(device)

        self.betas = betas          # (T, )
        self.alphas = 1.0 - betas    # (T, )

        # [bar_a_0, bar_a_1 ... bar_a_{T-1}]
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # [1, bar_a_0, bar_a_1 ... bar_a_{T-2}]
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=betas.device), self.alphas_cumprod[:-1]], 
            dim=0
        )

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recip_1m_alphas_cumprod = torch.sqrt(1.0 / (1.0 - self.alphas_cumprod))


    @staticmethod
    def _extract(
        a: torch.Tensor, # (T,)
        t: torch.Tensor, # (B,)
    ): 
        a = a.to(t.device)
        
        out = a[t].float() # (B,)
        return out.view(-1, 1, 1, 1).to(t.device) # (B,1,1,1)

    # ==== Forward Process ====
    def add_noise(
        self, 
        x_0: torch.Tensor, 
        noise: torch.Tensor, 
        t: torch.Tensor
    ):
        """
        q(x_t | x_0)
        x_t = sqrt(bar_a(t)) * x_0 + sqrt(1 - bar_a(t)) * eps
        x_0, gaus_noise: (B, C, H, W)
        t: (B,) int 0 -> 999
        """
        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, t
        ) # (B, C, H, W)
        sqrt_1m_alphas_cumprod_t = self._extract(
            self.sqrt_1m_alphas_cumprod, t
        )
        return sqrt_alphas_cumprod_t * x_0 + sqrt_1m_alphas_cumprod_t * noise
    
    # ==== Reverse Process ====
    def step(self, x_t, noise_pred, t):
        betas_t = self._extract(self.betas, t)             
        alphas_t = self._extract(self.alphas, t)           
        sqrt_recip_1m_alphas_cumprod_t = self._extract(self.sqrt_recip_1m_alphas_cumprod, t)

        # x_{t-1} = 1/sqrt(alpha_t) * ( x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps )
        mean = (
            (1.0 / torch.sqrt(alphas_t)) *
            (x_t - (betas_t * sqrt_recip_1m_alphas_cumprod_t) * noise_pred)
        )

        noise = torch.randn_like(x_t)

        nonzero_mask = (t > 0).float().view(-1,1,1,1)
        sigma_t = torch.sqrt(betas_t)

        x_prev = mean + nonzero_mask * sigma_t * noise
        return x_prev