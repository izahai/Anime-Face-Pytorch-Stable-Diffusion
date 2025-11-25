import argparse

from trainer.ldm_trainer import LatentDiffusionTrainer
from data.download_anime_faces import download_anime_faces
from tools.utils import load_yaml, merge_args_with_yaml, build_dataloaders
from models.ldm import LatentDiffusion
from models.ddpm import DDPMScheduler
from models.vae import AutoEncoder
from models.unet import UNet


def add_addition_argument(parser: argparse.ArgumentParser):
    parser.add_argument("--config", type=str, default="configs/vae.yaml", help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume training")

def main():
    parser = argparse.ArgumentParser("VAE Trainer")
    add_addition_argument(parser)
    args = parser.parse_args()

    yaml_cfg = load_yaml(args.config)
    args = merge_args_with_yaml(args, yaml_cfg)
    
    download_anime_faces()

    # Build loaders
    train_loader, val_loader = build_dataloaders(args)

    # Model
    unet = UNet()
    vae = AutoEncoder()
    scheduler = DDPMScheduler()

    model = LatentDiffusion(unet, vae, scheduler)

    # Train
    trainer = LatentDiffusionTrainer(args, train_loader, val_loader, model)
    trainer.train(resume_path=args.resume)

if __name__ == "__main__":
    main()