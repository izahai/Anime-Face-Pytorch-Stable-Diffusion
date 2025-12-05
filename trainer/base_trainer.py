# trainer/base_trainer.py

import os
import torch
from abc import ABC, abstractmethod
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()


class Trainer(ABC):
    """
    Generic base trainer with:
    - device setup
    - out_dir / samples / checkpoints dirs
    - HF push config
    - epoch / global_step / best_val_loss tracking
    - generic (save|load)_checkpoint helpers
    - generic push_to_hf helper

    Subclasses MUST:
      - set self.model and self.optimizer in their __init__
      - implement train() and validate()
    """

    def __init__(self, args, train_loader, val_loader):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dataloaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Output dirs
        self.out_dir = args.out_dir
        os.makedirs(os.path.join(self.out_dir, "samples"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "checkpoints"), exist_ok=True)

        # HuggingFace config
        self.push_to_hf_enabled = getattr(args, "push_to_hf", False)
        if self.push_to_hf_enabled:
            self.hf_username = os.getenv("HF_USERNAME", "")
            self.hf_token = os.getenv("HF_TOKEN")

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        # To be set by subclass
        self.model = None
        self.optimizer = None

    # ------------------------------------------------------------------
    # Abstract API for subclasses
    # ------------------------------------------------------------------
    @abstractmethod
    def train(self, resume_path=None):
        """Main training loop."""
        raise NotImplementedError

    @abstractmethod
    def validate(self, *args, **kwargs):
        """Run validation, return at least val_loss."""
        raise NotImplementedError
    
    def count_params(self, model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    # ------------------------------------------------------------------
    # Checkpoint utilities
    # ------------------------------------------------------------------
    def _build_state_dict(self):
        if self.model is None or self.optimizer is None:
            raise ValueError("model and optimizer must be set in subclass before saving.")
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
        }

    def save_checkpoint(self, ckpt_path: str):
        """Generic checkpoint saver; subclasses can call this with any path."""
        state = self._build_state_dict()
        torch.save(state, ckpt_path)
        print(f"[SAVE] Saved checkpoint: {ckpt_path}")

    def load_checkpoint(self, ckpt_path: str, strict: bool = True):
        """
        Generic checkpoint loader; expects keys:
        - 'model', 'optimizer', 'epoch', 'global_step'
        """
        print(f"[LOAD] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)

        if self.model is None or self.optimizer is None:
            raise ValueError("model and optimizer must be set in subclass before loading.")

        self.model.load_state_dict(ckpt["model"], strict=strict)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epoch = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)

        print(f"[RESUME] from epoch {self.epoch}, step {self.global_step}")

    # ------------------------------------------------------------------
    # HF push helper
    # ------------------------------------------------------------------
    def push_to_hf(self, ckpt_path: str, repo_suffix: str):
        """
        Generic HF uploader.
        - repo_suffix: e.g. 'anime_face_vae_epoch' or 'latent_diffusion_model'
        """
        if not self.push_to_hf_enabled:
            print("[HF] push_to_hf is disabled (args.push_to_hf=False).")
            return

        if not getattr(self, "hf_username", None) or not getattr(self, "hf_token", None):
            print("[HF] HF_USERNAME or HF_TOKEN not set; skipping upload.")
            return

        api = HfApi(token=self.hf_token)
        repo_id = f"{self.hf_username}/{repo_suffix}"

        print(f"[HF] Uploading {ckpt_path} to {repo_id}...")

        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        api.upload_file(
            path_or_fileobj=ckpt_path,
            path_in_repo=f"epoch{self.epoch}.pt",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Update best checkpoint at epoch {self.epoch}",
        )

        print(f"[HF] Uploaded to https://huggingface.co/{repo_id}")

        loss_json_path = os.path.join(self.args.out_dir, "logs", "losses.json")

        if os.path.exists(loss_json_path):
            print(f"[HF] Uploading losses.json...")

            api.upload_file(
                path_or_fileobj=loss_json_path,
                path_in_repo="losses.json",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Update losses at epoch {self.epoch}",
            )
            print(f"[HF] Uploaded losses.json ✅")
        else:
            print("[HF] losses.json not found, skipping.")