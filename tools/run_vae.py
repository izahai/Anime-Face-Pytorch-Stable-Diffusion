import torch
import argparse
from torchvision.utils import save_image
from models.vae import AutoEncoder

def load_vae(ckpt_path, image_size=64, z_ch=4, base_ch=64, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoEncoder(
        in_ch=3,
        base_ch=base_ch,
        z_ch=z_ch,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()

    latent_size = image_size // 4

    return model, device, latent_size, z_ch

def sample_latent(n, z_ch, latent_size, device):
    return torch.randn(
        n, z_ch, latent_size, latent_size,
        device=device
    )

def vae_generate(model, z):
    with torch.no_grad():
        return model.decode(z)
    
def save_single_sample(model, device, z_ch, latent_size, out_path="sample.png"):
    z = sample_latent(1, z_ch, latent_size, device)
    img = vae_generate(model, z)[0]

    img = (img + 1) * 0.5
    img = img.clamp(0, 1)

    save_image(img, out_path)
    print(f"[VAE] Saved sample to {out_path}")

def save_sample_grid(model, device, z_ch, latent_size, n=10, nrow=5, out_path="grid.png"):
    z = sample_latent(n, z_ch, latent_size, device)
    imgs = vae_generate(model, z)

    imgs = (imgs + 1) * 0.5
    imgs = imgs.clamp(0, 1)

    save_image(imgs, out_path, nrow=nrow)
    print(f"[VAE] Saved grid of {n} samples to {out_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="VAE Sampling Script")

    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to VAE checkpoint (.pt file)")

    parser.add_argument("--out", type=str, default="sample.png",
                        help="Output image path")

    parser.add_argument("--grid", action="store_true",
                        help="Generate a grid of samples instead of a single image")

    parser.add_argument("--n", type=int, default=10,
                        help="Number of images to sample when using --grid")

    parser.add_argument("--nrow", type=int, default=5,
                        help="How many images per row in grid")

    parser.add_argument("--image_size", type=int, default=64,
                        help="Image resolution VAE was trained on")

    parser.add_argument("--z_ch", type=int, default=4,
                        help="Latent channels of the VAE")

    parser.add_argument("--base_ch", type=int, default=64,
                        help="Base channel count of the VAE")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    model, device, latent_size, z_ch = load_vae(
        ckpt_path=args.ckpt,
        image_size=args.image_size,
        z_ch=args.z_ch,
        base_ch=args.base_ch,
        device=None
    )
    
    if args.grid:
        save_sample_grid(
            model=model,
            device=device,
            z_ch=z_ch,
            latent_size=latent_size,
            n=args.n,
            nrow=args.nrow,
            out_path=args.out
        )
    else:
        save_single_sample(
            model=model,
            device=device,
            z_ch=z_ch,
            latent_size=latent_size,
            out_path=args.out
        )