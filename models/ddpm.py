# models/diffusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class NoiseScheduler:
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
        out = a[t].float() # (B,)
        return out.view(-1, 1, 1, 1).to(t.device) # (B,1,1,1)

    # ==== Forward Process ====
    def add_noise(
        self, 
        x_0: torch.Tensor, 
        gaus_noise: torch.Tensor, 
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
        return sqrt_alphas_cumprod_t * x_0 + sqrt_1m_alphas_cumprod_t * gaus_noise

    # def get_x0_by_xT_eps(
    #     self, x_t: torch.Tensor,
    #     t: torch.Tensor, 
    #     eps: torch.Tensor
    # ):
    #     """
    #     x_t = sqrt(bar_a(t)) * x_0 + sqrt(1 - bar_a(t)) * eps
    #     --> x_0 = x_t / sqrt(bar_a(t)) - sqrt(1 - bar_a(t)) * eps
    #     x_t, eps: (B, C, H, W)
    #     t: (B,)
    #     """
    #     sqrt_recip_alphas_cumprod_t = self._extract(
    #         self.sqrt_recip_alphas_cumprod, t
    #     ) # (B, C, H, W)
    #     sqrt_1m_alphas_cumprod_t = self._extract(
    #         self.sqrt_1m_alphas_cumprod, t
    #     )
    #     return sqrt_recip_alphas_cumprod_t * x_t - sqrt_1m_alphas_cumprod_t * eps
        
    # def q_posterior_mean(
    #     self, x_start: torch.Tensor, 
    #     x_t: torch.Tensor, 
    #     t: torch.Tensor
    # ):
    #     pass

    # def step(
    #     self,
    #     model_output: torch.Tensor,
    #     t: torch.Tensor,
    #     x_t: torch.Tensor,
    #     prediction_type: str = "epsilon",
    # ):
    #     pass


class DDPM(nn.Module):
    def __init__(self, model: nn.Module, noise_scheduler: NoiseScheduler):
        super().__init__()
        self.model = model
        self.noise_scheduler = noise_scheduler

    def forward(self, x_0):
        device = x_0.device
        b = x_0.size(0)

        t = torch.randint(
            low=0,
            high=self.noise_scheduler.num_steps,
            size=(b,),
            device=device,
            dtype=torch.long,
        )

        gaus_noise = torch.randn_like(x_0)

        x_t =  self.noise_scheduler.add_noise(x_0, gaus_noise, t)

        pred_noise = self.model(x_t, t)

        return F.mse_loss(pred_noise, gaus_noise)