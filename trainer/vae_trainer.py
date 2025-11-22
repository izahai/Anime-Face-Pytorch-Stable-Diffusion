import os
import torch
from torchvision.utils import save_image
from tqdm import tqdm
from torch import amp
from huggingface_hub import HfApi
from dotenv import load_dotenv
load_dotenv()

from losses.vae_loss import vae_loss
from models.vae import AutoEncoder

class VAETrainer:
    def __init__(self, args, train_loader, val_loader, model=None):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        os.makedirs(os.path.join(args.out_dir, "samples"), exist_ok=True)
        os.makedirs(os.path.join(args.out_dir, "checkpoints"), exist_ok=True)

        self.push_to_hf_enabled = getattr(args, "push_to_hf", False)
        if self.push_to_hf_enabled:
            self.hf_home = os.getenv("HF_HOME", "./hf_cache")
            self.hf_token = os.getenv("HF_TOKEN")
        
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.model = model or AutoEncoder(
            in_ch=3,
            base_ch=args.base_ch,
            z_ch=args.z_ch,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)

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
        save_path = f'{self.args.out_dir}/samples/recon_epoch_{self.epoch}.png'
        save_image(grid, save_path, nrow=x.shape[0])

        print(f"[VAE] Saved reconstruction: {save_path}")
        self.model.train()

    def save_checkpoint(self):
        ckpt_path = f'{self.args.out_dir}/checkpoints/vae_epoch_{self.epoch}.pt'
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
        if resume_path:
            self.load_checkpoint(resume_path)

        scaler = amp.GradScaler(self.device)

        for ep in range(self.epoch, self.args.num_epochs):
            self.epoch = ep
            pbar = tqdm(self.train_loader, desc=f"Epoch {ep}")

            for x in pbar:
                x = x.to(self.device)

                with amp.autocast(self.device):
                    recon, mu, logvar = self.model(x)
                    loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, beta=self.args.beta)

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

            if (ep + 1) % self.args.log_every == 0:
                self.save_reconstruction(x[:4])

            if (ep + 1) % self.args.save_every == 0:
                self.save_checkpoint()

            # ---- Validation ----
            if (ep + 1) % self.args.val_every == 0:
                val_loss, _, _ = self.validate(self.val_loader)

                # Save best checkpoint
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_ckpt = f'{self.args.out_dir}/checkpoints/best_vae_epoch_{self.epoch}.pt'
                    torch.save({
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "epoch": self.epoch,
                        "global_step": self.global_step
                    }, best_ckpt)
                    print(f"[VAE] New BEST checkpoint saved: {best_ckpt}")

                if self.args.push_to_hf:
                    self.push_to_hf(best_ckpt)

        print("[VAE] Training complete!")

    def validate(self, val_loader):
        """Runs one full validation epoch and returns avg loss metrics."""
        self.model.eval()
        total_loss, total_recon_loss, total_kl_loss = 0, 0, 0
        count = 0

        with torch.no_grad():
            for x in val_loader:
                x = x.to(self.device)

                recon, mu, logvar = self.model(x)
                loss, recon_loss, kl_loss = vae_loss(
                    recon, x, mu, logvar, beta=self.args.beta
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
    
    def push_to_hf(self, ckpt_path):
        api = HfApi(token=self.hf_token)
        repo_id = f"{self.hf_home}/anime_face_vae_epoch"

        print(f"[HF] Uploading {ckpt_path} to {repo_id}...")

        api.upload_file(
            path_or_fileobj=ckpt_path,
            path_in_repo=f"epoch{self.epoch}.pt",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Update best checkpoint at epoch {self.epoch}"
        )

        print(f"[HF] Uploaded to https://huggingface.co/{repo_id}")