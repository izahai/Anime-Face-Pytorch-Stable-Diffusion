import os
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from losses.vae_loss import vae_loss
from models.vae import AutoEncoder

class VAETrainer:
    def __init__(self, cfg, train_loader, val_loader, model=None):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.train_loader = train_loader
        self.val_loader = val_loader
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
        self.best_val_loss = float("inf")

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
            pbar = tqdm(self.train_loader, desc=f"Epoch {ep}")

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

            # ---- End of Epoch ----
            if (ep + 1) % self.cfg["log_every"] == 0:
                self.save_reconstruction(x[:4])

            if (ep + 1) % self.cfg["save_every"] == 0:
                self.save_checkpoint()

            # ---- Validation ----
            if (ep + 1) % self.cfg["val_every"] == 0:
                val_loss, val_recon, val_kl = self.validate(self.val_loader)

                # Save best checkpoint
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_ckpt = f'{self.cfg["out_dir"]}/checkpoints/best.pt'
                    torch.save({
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "epoch": self.epoch,
                        "global_step": self.global_step
                    }, best_ckpt)
                    print(f"[VAE] New BEST checkpoint saved: {best_ckpt}")

        print("[VAE] Training complete!")

    def validate(self, val_loader):
        """Runs one full validation epoch and returns avg loss metrics."""
        self.model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        count = 0

        with torch.no_grad():
            for x in val_loader:
                x = x.to(self.device)

                recon, mu, logvar = self.model(x)
                loss, recon_loss, kl_loss = vae_loss(
                    recon, x, mu, logvar, beta=self.cfg["beta"]
                )

                total_loss += loss.item() * x.size(0)
                total_recon_loss += recon_loss.item() * x.size(0)
                total_kl_loss += kl_loss.item() * x.size(0)
                count += x.size(0)

        avg_loss = total_loss / count
        avg_recon = total_recon_loss / count
        avg_kl = total_kl_loss / count

        print(
            f"[VAL] Epoch {self.epoch} | "
            f"loss={avg_loss:.4f} | recon={avg_recon:.4f} | KL={avg_kl:.4f}"
        )

        self.model.train()
        return avg_loss, avg_recon, avg_kl

 
