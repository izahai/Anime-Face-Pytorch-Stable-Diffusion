# trainer/vae_gan_trainer.py

import torch
import torch.nn.functional as F
from torch import amp
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import os
import json

from trainer.base_trainer import Trainer
from models.vae import AutoEncoder
from models.discriminator import Discriminator
from losses.vae_loss import vae_gan_lpips_charbonnier_loss, charbonnier_loss
import lpips


class VAEGANTrainer(Trainer):
    """
    Full VAE + GAN + LPIPS trainer.
    - Generator: AutoEncoder (VAE)
    - Discriminator: PatchGAN
    """

    def __init__(self, args, train_loader, val_loader, model=None, lpips_model=None):
        super().__init__(args, train_loader, val_loader)

        # -------------------------
        # Setup Generator (VAE)
        # -------------------------
        self.vae = model or AutoEncoder(
            in_ch=3,
            base_ch=args.base_ch,
            z_ch=args.z_ch,
            factor=args.factor,
            num_head=args.num_head,
        ).to(self.device)

        self.optimizer_g = torch.optim.Adam(
            self.vae.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
        )

        # -------------------------
        # Setup Discriminator
        # -------------------------
        self.discriminator = Discriminator(
            in_ch=3,
            base_ch=args.disc_ch,
        ).to(self.device)

        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=args.learning_rate * args.d_lr_scale,
            betas=(args.adam_beta1, args.adam_beta2),
        )

        # -------------------------
        # LPIPS
        # -------------------------
        if lpips_model is None:
            self.lpips_model = lpips.LPIPS(net="vgg")
        else:
            self.lpips_model = lpips_model

        self.lpips_model = self.lpips_model.to(self.device)
        self.lpips_model.eval()

        g_total, g_trainable = self.count_params(self.vae)
        d_total, d_trainable = self.count_params(self.discriminator)

        self.loss_meter = {
            "g": 0.0,
            "d": 0.0,
            "recon": 0.0,
            "lpips": 0.0,
            "kl": 0.0,
            "steps": 0,
        }
        self.loss_log_path = os.path.join(self.args.out_dir, "logs", "losses.json")
        os.makedirs(os.path.dirname(self.loss_log_path), exist_ok=True)

        if not os.path.exists(self.loss_log_path):
            with open(self.loss_log_path, "w") as f:
                json.dump({}, f, indent=4)

        print("=" * 60)
        print("[MODEL PARAMS]")
        print(f"Generator (VAE):")
        print(f"  Total params:     {g_total:,}")
        print(f"  Trainable params: {g_trainable:,}")
        print(f"Discriminator:")
        print(f"  Total params:     {d_total:,}")
        print(f"  Trainable params: {d_trainable:,}")
        print(f"LPIPS:")
        lp_total = sum(p.numel() for p in self.lpips_model.parameters())
        print(f"  Total params:     {lp_total:,} (frozen)")
        print("=" * 60)

    # ----------------------------------------------------------------
    # Save reconstructions (same as VAETrainer)
    # ----------------------------------------------------------------
    def save_reconstruction(self, x=None, num_samples=8, nrow=None):
        self.vae.eval()

        if x is None:
            for batch in self.train_loader:
                x = batch.to(self.device)
                break

        x = x[:num_samples]
        with torch.no_grad():
            recon, _, _ = self.vae(x)

        x_vis = (x + 1) * 0.5
        recon_vis = (recon + 1) * 0.5

        grid = torch.cat([x_vis, recon_vis], dim=0)
        grid = make_grid(grid, nrow=num_samples if nrow is None else nrow)

        save_path = f"{self.args.out_dir}/samples/recon_epoch_{self.epoch}.png"
        save_image(grid, save_path)
        print(f"[SAMPLE] Saved reconstruction: {save_path}")

        self.vae.train()

    # ----------------------------------------------------------------
    # Training Loop
    # ----------------------------------------------------------------
    def train(self, resume_path=None):
        if resume_path:
            self.load_checkpoint(resume_path)  # inherited from base class

        scaler = amp.GradScaler(enabled=(self.device.type == "cuda"))

        for ep in range(self.epoch, self.args.num_epochs):
            self.epoch = ep
            pbar = tqdm(self.train_loader, desc=f"Epoch {ep}")

            for x in pbar:
                x = x.to(self.device)

                # ============================================================
                # 1. Train Generator (VAE)
                # ============================================================
                with amp.autocast(device_type=self.device.type):
                    recon, mu, logvar = self.vae(x)

                    disc_fake = self.discriminator(recon)

                    g_loss, logs = vae_gan_lpips_charbonnier_loss(
                        recon=recon,
                        x=x,
                        mu=mu,
                        logvar=logvar,
                        disc_fake=disc_fake,
                        lpips_model=self.lpips_model,
                        beta_kl=self.args.kl_beta,
                        w_charb=self.args.w_charb,
                        w_lpips=self.args.w_lpips,
                        w_gan=self.args.w_gan,
                    )

                self.optimizer_g.zero_grad()
                scaler.scale(g_loss).backward()
                scaler.step(self.optimizer_g)

                # ============================================================
                # 2. Train Discriminator
                # ============================================================
                with amp.autocast(device_type=self.device.type):
                    # Real images
                    real_logits = self.discriminator(x)
                    d_real_loss = F.softplus(-real_logits).mean()

                    # Fake images (detach so G is not updated)
                    fake_logits = self.discriminator(recon.detach())
                    d_fake_loss = F.softplus(fake_logits).mean()

                    d_loss = d_real_loss + d_fake_loss

                self.optimizer_d.zero_grad()
                scaler.scale(d_loss).backward()
                scaler.step(self.optimizer_d)

                scaler.update()
                self.global_step += 1
                self.accum_loss(logs)

                pbar.set_postfix({
                    "G_loss": f"{logs["g"]:.4f}",
                    "D_loss": f"{logs["d"]:.4f}",
                    "recon": f"{logs['recon']:.4f}",
                    "lpips": f"{logs['lpips']:.4f}",
                    "kl": f"{logs['kl']:.4f}"
                })

                if self.args.test_pipeline == True:
                    break

            # ---------------------------
            # Logging
            # ---------------------------
            if (ep + 1) % self.args.log_every == 0:
                self.save_reconstruction(num_samples=16)

            # ---------------------------
            # Save checkpoint
            # ---------------------------
            if (ep + 1) % self.args.save_every == 0:
                ckpt_path = f"{self.args.out_dir}/checkpoints/vaegan_epoch_{self.epoch}.pt"
                self.save_checkpoint(ckpt_path)

            # ---------------------------
            # Validation
            # ---------------------------
            if (ep + 1) % self.args.val_every == 0:
                avg_val_loss, avg_recon, avg_kl_loss = self.validate()  # custom validate below
                train_losses = self.log_train_loss()
                self.log_val_loss(avg_val_loss, avg_recon, avg_kl_loss)

                val_losses = {
                    "total": avg_val_loss,
                    "recon": avg_recon,
                    "kl": avg_kl_loss,
                }

                self.save_losses_to_json(train_losses, val_losses)
                
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    best_path = f"{self.args.out_dir}/checkpoints/best_vaegan_epoch_{self.epoch}.pt"
                    self.save_checkpoint(best_path)
                    print(f"[BEST] Saved best model at: {best_path}")

                    if self.push_to_hf_enabled:
                        self.push_to_hf(best_path, repo_suffix="vaegan_model")

        print("[VAE-GAN] Training complete!")

    # ----------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------
    @torch.no_grad()
    def validate(self):
        self.vae.eval()

        total_loss = 0
        total_recon = 0
        total_kl = 0
        count = 0

        for x in self.val_loader:
            x = x.to(self.device)
            recon, mu, logvar = self.vae(x)

            # no GAN during validation
            recon_loss = charbonnier_loss(recon, x)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl = self.args.kl_beta * kl / x.numel()
            loss = recon_loss + kl

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl.item()
            count += 1

        avg_loss = total_loss / count
        avg_recon_loss = total_recon / count
        avg_kl = total_kl / count

        self.vae.train()
        return avg_loss, avg_recon_loss, avg_kl
    
    def accum_loss(self, logs):
        self.loss_meter["g"] += logs["g"]
        self.loss_meter["d"] += logs["d"]
        self.loss_meter["recon"] += logs["recon"]
        self.loss_meter["lpips"] += logs["lpips"]
        self.loss_meter["kl"] += logs["kl"]
        self.loss_meter["steps"] += 1

    
    def _build_state_dict(self):
        return {
            "vae": self.vae.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
            "epoch": self.epoch,
        }
    
    def log_train_loss(self):
        meter = self.loss_meter
        avg_meter = {k: v / max(meter["steps"], 1) for k, v in meter.items()}
        print("\n" + "=" * 90)
        print(f"🟢 EPOCH {self.epoch} SUMMARY")
        print("-" * 90)
        print(
            f"TRAIN | "
            f"G: {avg_meter['g']:.6f} | "
            f"D: {avg_meter['d']:.6f} | "
            f"Recon: {avg_meter['recon']:.6f} | "
            f"LPIPS: {avg_meter['lpips']:.6f} | "
            f"KL: {avg_meter['kl']:.6f}"
        )
        train_losses = {
            "g": avg_meter["g"],
            "d": avg_meter["d"],
            "recon": avg_meter["recon"],
            "lpips": avg_meter["lpips"],
            "kl": avg_meter["kl"],
        }
        print("=" * 90 + "\n")
        for k in meter:
            meter[k] = 0.0
        return train_losses

    def log_val_loss(self, avg_loss, avg_recon_loss, avg_kl):
        print("\n" + "=" * 90)
        print(f"🔵 EPOCH {self.epoch} VALIDATION SUMMARY")
        print("-" * 90)
        print(
            f"VAL | "
            f"Total: {avg_loss:.6f} | "
            f"Recon: {avg_recon_loss:.6f}"
            f"KL: {avg_kl:.6f}"
        )
        print("=" * 90 + "\n")

    def save_losses_to_json(self, train_losses, val_losses=None):
        with open(self.loss_log_path, "r") as f:
            data = json.load(f)

        data[str(self.epoch)] = {
            "train": train_losses,
            "val": val_losses
        }

        with open(self.loss_log_path, "w") as f:
            json.dump(data, f, indent=4)


        


