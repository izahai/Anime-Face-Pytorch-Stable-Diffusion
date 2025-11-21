import torch
import torch.nn.functional as F

def vae_loss(recon, x, mu, logvar, beta=1e-6):
    recon_loss = F.mse_loss(recon, x)

    # KL = -0.5 * sum(1 + logvar )
    kl = -0.5 * torch.sum(1+logvar - mu.pow(2) - logvar.exp())
    kl = kl / (x.size(0) * x.size(2) * x.size(3))

    total_loss = recon_loss + beta*kl
    return total_loss, recon_loss, kl