import torch
import argparse
import random
from torchvision import transforms
from PIL import Image
import os
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

def save_random_reconstruction_grid(
    model,
    folder,
    image_size,
    recon_n=10,
    nrow=5,
    out_path="recon_grid.png"
):
    """
    Randomly sample N images from a folder, reconstruct using the VAE,
    and save a grid of [original | reconstruction].

    Args:
        model: AutoEncoder
        folder: path to folder containing images
        image_size: resize/crop size
        recon_n: number of random images
        nrow: images per row *for the merged grid*
        out_path: save file
    """
    device = next(model.parameters()).device
    model.eval()

    # --- Load files ---
    files = [f for f in os.listdir(folder) if f.lower().endswith(("png", "jpg", "jpeg"))]
    assert len(files) > 0, "No image files found in folder"

    # random select recon_n images
    chosen = random.sample(files, k=min(recon_n, len(files)))

    tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),     # [0,1]
    ])

    imgs = []
    for f in chosen:
        img = Image.open(os.path.join(folder, f)).convert("RGB")
        imgs.append(tf(img))

    imgs = torch.stack(imgs, dim=0)   # (B,3,H,W)

    # ---- Convert to [-1,1] ----
    imgs_norm = imgs * 2 - 1
    imgs_norm = imgs_norm.to(device)

    with torch.no_grad():
        z = model.encode_to_z(imgs_norm)
        recons = model.decode(z)

    # back to [0,1]
    imgs_vis = (imgs_norm + 1) * 0.5
    recons_vis = (recons + 1) * 0.5

    # merge horizontally (B, 3, H, W*2)
    merged = torch.cat([imgs_vis, recons_vis], dim=3)
    merged = merged.clamp(0, 1)

    save_image(merged, out_path, nrow=nrow)
    print(f"[VAE] Saved random reconstruction grid → {out_path}")


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
    
    parser.add_argument("--recon", action="store_true",
                    help="Run random reconstruction mode")

    parser.add_argument("--recon_n", type=int, default=10,
                        help="Number of random images for reconstruction")

    parser.add_argument("--input_folder", type=str, default=None,
                        help="Folder containing images for reconstruction")


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

    if args.recon:
        if args.input_folder is None:
            raise ValueError("You must provide --input_folder for reconstruction.")

        save_random_reconstruction_grid(
            model=model,
            folder=args.input_folder,
            image_size=args.image_size,
            recon_n=args.recon_n,
            nrow=args.nrow,
            out_path=args.out
        )
        exit()

    
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