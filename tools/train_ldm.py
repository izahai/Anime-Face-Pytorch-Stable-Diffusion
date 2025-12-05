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
    
    image_dir, meta_path = download_anime_faces(
                                args.image_size, 
                                args.dataset_name,
                                args.metadata_name
                            )

    # Build loaders
    train_loader, val_loader = build_dataloaders(args, image_dir, meta_path)


    # Model
    unet = UNet(in_ch=args.z_ch, base_ch=args.base_ch, num_head=args.num_head)
    vae = AutoEncoder(base_ch=args.base_ch, z_ch=args.z_ch, num_head=args.num_head)
    scheduler = DDPMScheduler()

    model = LatentDiffusion(unet, vae, scheduler)

    # Train
    trainer = LatentDiffusionTrainer(args, train_loader, val_loader, model)
    trainer.train(resume_path=args.resume)

if __name__ == "__main__":
    main()