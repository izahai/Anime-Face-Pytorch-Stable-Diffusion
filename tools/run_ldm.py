# tools/sample_ldm.py

import argparse
import torch
from torchvision.utils import save_image

from models.ldm import LatentDiffusion
from models.unet import UNet          # make sure this matches your unet class name
from models.vae import AutoEncoder
from models.ddpm import DDPMScheduler


# ------------------------------
# Helpers to load submodules
# ------------------------------
def load_vae(ckpt_path, image_size=64, z_ch=4, base_ch=64, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    vae = AutoEncoder(
        in_ch=3,
        base_ch=base_ch,
        z_ch=z_ch,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    if "model" in state:        # handle dicts like {"model": state_dict}
        state = state["model"]
    vae.load_state_dict(state)
    vae.eval()

    latent_size = image_size // 4  # because your VAE downsamples by 4
    return vae, device, latent_size


def load_unet(ckpt_path, z_ch=4, base_ch=64, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    unet = UNet(
        in_ch=z_ch,
        base_ch=base_ch,
        ch_mult=(1, 2, 4),
        t_dim=256,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)

    # Case 1: inside {"model": ...}
    if "model" in state:
        state = state["model"]

    # Case 2: UNet is stored inside "unet"
    if "unet" in state:
        state = state["unet"]

    # ---- FIX HERE: strip "unet." prefix if present ----
    cleaned_state = {}
    for k, v in state.items():
        if k.startswith("unet."):
            cleaned_state[k[len("unet."):]] = v
        else:
            cleaned_state[k] = v

    missing, unexpected = unet.load_state_dict(cleaned_state, strict=False)
    
    print("Loaded UNet with:")
    print("  Missing keys:    ", missing)
    print("  Unexpected keys: ", unexpected)

    unet.eval()
    return unet



def build_scheduler(num_steps=1000, beta_start=1e-4, beta_end=2e-2, device=None):
    sched = DDPMScheduler(
        num_steps=num_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        device=device,
    )

    # IMPORTANT: ensure all buffers are on the right device
    if device is not None:
        sched.betas = sched.betas.to(device)
        sched.alphas = sched.alphas.to(device)
        sched.alphas_cumprod = sched.alphas_cumprod.to(device)
        sched.alphas_cumprod_prev = sched.alphas_cumprod_prev.to(device)
        if hasattr(sched, "sqrt_alphas_cumprod"):
            sched.sqrt_alphas_cumprod = sched.sqrt_alphas_cumprod.to(device)
        if hasattr(sched, "sqrt_one_minus_alphas_cumprod"):
            sched.sqrt_one_minus_alphas_cumprod = \
                sched.sqrt_one_minus_alphas_cumprod.to(device)
        if hasattr(sched, "posterior_variance"):
            sched.posterior_variance = sched.posterior_variance.to(device)

    return sched


# ------------------------------
# Sampling utilities
# ------------------------------
@torch.no_grad()
def ldm_sample(ldm, num_samples, latent_size, z_ch, device, out_path, nrow=None):
    nrow = nrow or int(num_samples ** 0.5)

    imgs = ldm.sample(
        num_samples=num_samples,
        latent_shape=(z_ch, latent_size, latent_size),
        device=device,
    )

    # VAE outputs are in [-1, 1], convert to [0, 1]
    imgs = (imgs + 1) * 0.5
    imgs = imgs.clamp(0, 1)

    save_image(imgs, out_path, nrow=nrow)
    print(f"[LDM] Saved {num_samples} samples to {out_path}")


# ------------------------------
# CLI
# ------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Latent Diffusion Sampling")

    parser.add_argument("--vae_ckpt", type=str, required=True,
                        help="Path to VAE checkpoint (.pt)")

    parser.add_argument("--unet_ckpt", type=str, required=True,
                        help="Path to UNet/diffusion checkpoint (.pt)")

    parser.add_argument("--out", type=str, default="ldm_samples.png",
                        help="Output image path")

    parser.add_argument("--num_samples", type=int, default=16,
                        help="Number of images to sample")

    parser.add_argument("--nrow", type=int, default=4,
                        help="Images per row in the grid")

    parser.add_argument("--image_size", type=int, default=64,
                        help="Image resolution (VAE training size)")

    parser.add_argument("--z_ch", type=int, default=4,
                        help="Latent channels used in VAE")

    parser.add_argument("--base_ch", type=int, default=64,
                        help="Base channel count for both VAE & UNet")

    parser.add_argument("--num_steps", type=int, default=1000,
                        help="Number of diffusion steps (same as training)")

    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=2e-2)

    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load VAE
    vae, device, latent_size = load_vae(
        ckpt_path=args.vae_ckpt,
        image_size=args.image_size,
        z_ch=args.z_ch,
        base_ch=args.base_ch,
        device=device,
    )

    # 2. Load UNet
    unet = load_unet(
        ckpt_path=args.unet_ckpt,
        z_ch=args.z_ch,
        base_ch=args.base_ch,
        device=device,
    )

    # 3. Build scheduler
    scheduler = build_scheduler(
        num_steps=args.num_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )

    # 4. Build full Latent Diffusion model
    ldm = LatentDiffusion(
        unet=unet,
        vae=vae,
        scheduler=scheduler,
    ).to(device)
    ldm.eval()

    # 5. Sample & save
    ldm_sample(
        ldm=ldm,
        num_samples=args.num_samples,
        latent_size=latent_size,
        z_ch=args.z_ch,
        device=device,
        out_path=args.out,
        nrow=args.nrow,
    )


if __name__ == "__main__":
    main()