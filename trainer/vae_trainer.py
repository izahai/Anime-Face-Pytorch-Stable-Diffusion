import os
import torch
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from torch import amp
from huggingface_hub import HfApi
import torch
from dotenv import load_dotenv
load_dotenv()

from losses.vae_loss import vae_loss_L2, vae_gan_lpips_charbonnier_loss
from models.vae import AutoEncoder
from trainer.base_trainer import Trainer

class VAETrainer(Trainer):
    def __init__(self, args, train_loader, val_loader, model=None):
        super().__init__(args, train_loader, val_loader)

        self.model = model or AutoEncoder(
            in_ch=3,
            base_ch=args.base_ch,
            z_ch=args.z_ch,
            factor=args.factor,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.learning_rate,
        )

    def save_reconstruction(self, x=None, num_samples=8, nrow=None):
        """
        Saves a single image grid per epoch containing:
        - num_samples original images
        - num_samples reconstructions
        
        If x is None, it will grab one batch from train_loader automatically.
        """
        
        self.model.eval()
        
        # ---- If x not provided, fetch 1 batch ----
        if x is None:
            for batch in self.train_loader:
                x = batch.to(self.device)
                break  # Only need 1 batch
        
        # ---- Select only num_samples ----
        x = x[:num_samples].to(self.device)
        
        with torch.no_grad():
            recon, _, _ = self.model(x)

        # Denormalize [-1,1] → [0,1]
        x_vis = (x + 1) * 0.5
        recon_vis = (recon + 1) * 0.5

        # ---- Construct a grid ----
        # Stack like:
        # [x0, x1, x2, ...]
        # [r0, r1, r2, ...]
        grid = torch.cat([x_vis, recon_vis], dim=0)

        # nrow = num_samples when not provided
        if nrow is None:
            nrow = num_samples

        grid = make_grid(grid, nrow=nrow, padding=2)

        # ---- Save ----
        save_path = f"{self.args.out_dir}/samples/recon_epoch_{self.epoch}.png"
        save_image(grid, save_path)

        print(f"[SAMPLE] Saved reconstruction grid: {save_path}")
        self.model.train()

    def train(self, resume_path=None):
        # -------- Resume if provided --------
        if resume_path:
            self.load_checkpoint(resume_path)

        scaler = amp.GradScaler(enabled=(self.device.type == "cuda"))

        for ep in range(self.epoch, self.args.num_epochs):
            self.epoch = ep
            pbar = tqdm(self.train_loader, desc=f"Epoch {ep}")

            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            num_batches = 0
            for x in pbar:
                x = x.to(self.device)

                with amp.autocast(device_type=self.device.type):
                    recon, mu, logvar = self.model(x)
                    loss, recon_loss, kl_loss = vae_loss_L2(recon, x, mu, logvar, beta=self.args.kl_beta)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.global_step += 1

                pbar.set_postfix({
                    "loss": f"{loss:.4f}",
                    "recon": f"{recon_loss:.4f}",
                    "kl": f"{kl_loss:.4f}"
                })

                epoch_loss += loss.item()
                epoch_recon += recon_loss.item()
                epoch_kl += kl_loss.item()
                num_batches += 1

            if (ep + 1) % self.args.log_every == 0:
                self.save_reconstruction(num_samples=16)

            if (ep + 1) % self.args.save_every == 0:
                ckpt_path = f'{self.args.out_dir}/checkpoints/vae_epoch_{self.epoch}.pt'
                self.save_checkpoint(ckpt_path)
            
            # ---- Train Loss ----
            mean_loss = epoch_loss / num_batches
            mean_recon = epoch_recon / num_batches
            mean_kl = epoch_kl / num_batches

            print(
                f"[TRAIN] Epoch {self.epoch} | "
                f"loss={mean_loss:.4f} | recon={mean_recon:.4f} | KL={mean_kl:.4f}"
            )

            # ---- Validation ----
            if (ep + 1) % self.args.val_every == 0:
                val_loss, _, _ = self.validate(self.val_loader)

                # Save best checkpoint
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_ckpt_path = f'{self.args.out_dir}/checkpoints/best_vae_epoch_{self.epoch}.pt'
                    self.save_checkpoint(best_ckpt_path)
                    print(f"[BEST] new checkpoint saved: {best_ckpt_path}")

                    if self.args.push_to_hf:
                        self.push_to_hf(best_ckpt_path)

        print("[TRAINING COMPLETE] :D")

    def validate(self, val_loader):
        """Runs one full validation epoch and returns avg loss metrics."""
        self.model.eval()
        total_loss, total_recon_loss, total_kl_loss = 0, 0, 0
        count = 0

        with torch.no_grad():
            for x in val_loader:
                x = x.to(self.device)

                recon, mu, logvar = self.model(x)
                loss, recon_loss, kl_loss = vae_loss_L2(
                    recon, x, mu, logvar, beta=self.args.kl_beta
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