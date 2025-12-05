# trainers/ldm_trainer.py

import os
import torch
from torch import amp
from torchvision.utils import save_image
from tqdm import tqdm
from huggingface_hub import HfApi
from dotenv import load_dotenv

from trainer.base_trainer import Trainer

load_dotenv()

class LatentDiffusionTrainer(Trainer):
    def __init__(self, args, train_loader, val_loader, model=None):
        super().__init__(args, train_loader, val_loader)

        # ─── Model & Optimizer ───────────────────────────────────
        self.model = model.to(self.device)

        if args.pretrained_vae:
            print(f"[LDM] Loading pretrained VAE: {args.pretrained_vae}")
            vae_ckpt = torch.load(args.pretrained_vae, map_location=self.device)

            # allow both "model" key or raw state dict
            if "vae" in vae_ckpt:
                state_dict = vae_ckpt["vae"]
            elif "model" in vae_ckpt:
                state_dict = vae_ckpt["model"]
            else:
                state_dict = vae_ckpt

            self.model.vae.load_state_dict(state_dict, strict=True)
            print("[LDM] Pretrained VAE loaded successfully!")

        for p in self.model.vae.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(
            self.model.unet.parameters(),
            lr=args.learning_rate
        )

    # ======================================================================
    # Save sample images (sample from noise using model.sample)
    # ======================================================================
    @torch.no_grad()
    def save_sample(self, num=4, z_ch=4):
        self.model.eval()

        z = self.model.sample(num_samples=num, latent_shape=(z_ch, 16, 16), device=self.device)
        x = (z + 1) * 0.5  # de-normalize

        save_path = f"{self.args.out_dir}/samples/sample_epoch_{self.epoch}.png"
        save_image(x, save_path, nrow=num)

        print(f"[LDM] Saved samples: {save_path}")
        self.model.train()

    # ======================================================================
    # Save checkpoint
    # ======================================================================
    def save_checkpoint(self):
        ckpt_path = f"{self.args.out_dir}/checkpoints/ldm_epoch_{self.epoch}.pt"
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step
        }, ckpt_path)
        print(f"[LDM] Saved checkpoint: {ckpt_path}")

    # ======================================================================
    # Load checkpoint
    # ======================================================================
    def load_checkpoint(self, ckpt_path):
        print(f"[LDM] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epoch = ckpt["epoch"]
        self.global_step = ckpt["global_step"]

        print(f"[LDM] Resumed from epoch {self.epoch}, step {self.global_step}")

    # ======================================================================
    # Validation step (simple average loss)
    # ======================================================================
    @torch.no_grad()
    def validate(self):
        self.model.eval()

        total_loss = 0
        count = 0

        for imgs in self.val_loader:
            imgs = imgs.to(self.device)

            loss, pred_noise, true_noise, t = self.model(imgs)

            total_loss += loss.item() * imgs.size(0)
            count += imgs.size(0)

        avg_loss = total_loss / count

        print(f"[VAL] Epoch {self.epoch} | loss={avg_loss:.4f}")

        self.model.train()

        return avg_loss

    # ======================================================================
    # HuggingFace Push
    # ======================================================================
    def push_to_hf(self, ckpt_path):
        api = HfApi(token=self.hf_token)
        repo_id = f"{self.hf_username}/latent_diffusion_model"

        print(f"[HF] Uploading {ckpt_path} to {repo_id}...")

        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        api.upload_file(
            path_or_fileobj=ckpt_path,
            path_in_repo=f"epoch{self.epoch}.pt",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Update best checkpoint at epoch {self.epoch}"
        )
        print(f"[HF] Uploaded to https://huggingface.co/{repo_id}")

    # ======================================================================
    # Training Loop
    # ======================================================================
    def train(self, resume_path=None):
        # Resume training if needed
        if resume_path:
            self.load_checkpoint(resume_path)

        scaler = amp.GradScaler(enabled=(self.device.type == "cuda"))

        for ep in range(self.epoch, self.args.num_epochs):
            self.epoch = ep

            pbar = tqdm(self.train_loader, desc=f"Epoch {ep}")

            for imgs in pbar:
                imgs = imgs.to(self.device)

                with amp.autocast(device_type=self.device.type):
                    loss, pred_noise, true_noise, t = self.model(imgs)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                self.global_step += 1

                pbar.set_postfix({"loss": f"{loss:.4f}"})
                
                if self.args.test_pipeline:
                    break

            # ─── Logging & Sample Generation ──────────────────────────────
            if (ep + 1) % self.args.log_every == 0:
                self.save_sample(num=100, z_ch=self.args.z_ch)

            # ─── Save Checkpoint ───────────────────────────────────────────
            if (ep + 1) % self.args.save_every == 0:
                self.save_checkpoint()

            # ─── Validation ───────────────────────────────────────────────
            if (ep + 1) % self.args.val_every == 0:
                val_loss = self.validate()

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_ckpt = f"{self.args.out_dir}/checkpoints/best_ldm_epoch_{self.epoch}.pt"

                    torch.save({
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "epoch": self.epoch,
                        "global_step": self.global_step
                    }, best_ckpt)

                    print(f"[LDM] New BEST checkpoint saved: {best_ckpt}")

                    if self.push_to_hf_enabled:
                        self.push_to_hf(best_ckpt)

        print("[LDM] Training complete!")