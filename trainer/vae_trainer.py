import os
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from losses.vae_loss import vae_loss
from models.vae import AutoEncoder

class VAETrainer:
    def __init__(self, cfg, dataloader, model=None):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.dataloader = dataloader
        self.model = model or AutoEncoder(
            in_ch=3,
            base_ch=cfg["base_ch"],
            z_ch=cfg["z_ch"]
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg["learning_rate"]
        )

        self.epoch = 0
        self.global_step = 0

    def save_reconstruction(self, x):
        self.model.eval()
        with torch.no_grad():
            recon, _, _ = self.model(x)

        # De-normalize from [-1,1] → [0,1]
        x_vis = (x + 1) * 0.5
        recon_vis = (recon + 1) * 0.5

        grid = torch.cat([x_vis, recon_vis], dim=0)
        save_path = f'{self.cfg["out_dir"]}/samples/recon_epoch_{self.epoch}.png'
        save_image(grid, save_path, nrow=x.shape[0])

        print(f"[VAE] Saved reconstruction: {save_path}")
        self.model.train()

    def save_checkpoint(self):
        ckpt_path = f'{self.cfg["out_dir"]}/checkpoints/vae_epoch_{self.epoch}.pt'
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step
        }, ckpt_path)
        print(f"[VAE] Saved checkpoint: {ckpt_path}")


    def load_checkpoint(self, ckpt_path):
        print(f"[VAE] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epoch = ckpt["epoch"]
        self.global_step = ckpt["global_step"]
        print(f"[VAE] Resumed from epoch {self.epoch}, step {self.global_step}")


    def train(self, resume_path=None):
        # -------- Resume if provided --------
        if resume_path is not None and os.path.exists(resume_path):
            self.load_checkpoint(resume_path)

        num_epochs = self.cfg["num_epochs"]

        for ep in range(self.epoch, num_epochs):
            self.epoch = ep
            pbar = tqdm(self.dataloader, desc=f"Epoch {ep}")

            for x in pbar:
                x = x.to(self.device)

                recon, mu, logvar = self.model(x)
                loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, beta=self.cfg["beta"])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.global_step += 1

                pbar.set_postfix({
                    "loss": f"{loss:.4f}",
                    "recon": f"{recon_loss:.4f}",
                    "kl": f"{kl_loss:.4f}"
                })

            # End of epoch
            if (ep + 1) % self.cfg["log_every"] == 0:
                self.save_reconstruction(x[:4])

            if (ep + 1) % self.cfg["save_every"] == 0:
                self.save_checkpoint()

        print("[VAE] Training complete!")
