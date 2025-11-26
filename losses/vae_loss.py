import torch
import torch.nn.functional as F

def vae_loss_L2(recon, x, mu, logvar, beta=1e-6):
    recon_loss = F.mse_loss(recon, x)

    # KL = -0.5 * sum(1 + logvar)
    kl = -0.5 * torch.sum(1+logvar - mu.pow(2) - logvar.exp())
    kl = kl / (x.size(0) * x.size(2) * x.size(3))

    total_loss = recon_loss + beta*kl
    return total_loss, recon_loss, kl

def charbonnier_loss(x, y, eps=1e-6):
    return torch.mean(torch.sqrt((x - y)**2 + eps*eps))

def vae_gan_lpips_charbonnier_loss(
    recon, x, mu, logvar, disc_fake, lpips_model,
    beta_kl=1e-6, w_charb=1.0, w_lpips=1.0, w_gan=1.0
):
    l_charb = charbonnier_loss(recon, x)

    l_lpips = lpips_model(recon, x).mean()

    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl = kl / (x.size(0) * x.size(2) * x.size(3))

    l_gan = F.softplus(-disc_fake).mean() 

    total_loss = (
        w_charb * l_charb + 
        w_lpips * l_lpips +
        w_gan   * l_gan +
        beta_kl * kl
    )

    return total_loss, {
        "recon_charbonnier": l_charb,
        "lpips": l_lpips,
        "gan": l_gan,
        "kl": kl,
    }