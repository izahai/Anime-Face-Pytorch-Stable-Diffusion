import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from losses.vae_loss import vae_loss
from models.vae import AutoEncoder
from dataset.anime_face_ds import AnimeFolderDataset

class VAETrainer:
    def __init__(self, cfg):
        self.cfg = cfg

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        os.makedirs(cfg["out_dir"], exist_ok=True)
        os.makedirs(f'{cfg["out_dir"]}/samples', exist_ok=True)
        os.makedirs(f'{cfg["out_dir"]}/checkpoints', exist_ok=True)

        self.dataset = AnimeFolderDataset(
            folder=cfg["image_folder"],
            metadata_path=cfg("metadata_path"),
            image_size=cfg["image_size"],
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg.get("num_workers", 4)
        )

        self.model = AutoEncoder(
            in_ch=3,
            base_ch=cfg["base_ch"],
            z_ch=cfg["z_ch"]
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg["learning_rate"]
        )

        self.step = 0

    def save_reconstruction(self, x):
        self.model.eval()
        with torch.no_grad():
            recon, _, _ = self.model(x)

        # De-normalize from [-1,1] → [0,1]
        x_vis = (x + 1) * 0.5
        recon_vis = (recon + 1) * 0.5

        grid = torch.cat([x_vis, recon_vis], dim=0)
        save_path = f'{self.cfg["out_dir"]}/samples/recon_step_{self.step}.png'
        save_image(grid, save_path, nrow=x.shape[0])

        print(f"[VAE] Saved reconstruction: {save_path}")
        self.model.train()

    def save_checkpoint(self):
        ckpt_path = f'{self.cfg["out_dir"]}/checkpoints/vae_step_{self.step}.pt'
        torch.save(self.model.state_dict(), ckpt_path)
        print(f"[VAE] Saved checkpoint: {ckpt_path}")

    