import argparse

from trainer.vae_trainer import VAETrainer
from data.download_anime_faces import download_anime_faces
from tools.utils import load_yaml, merge_args_with_yaml, build_dataloaders


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

    # Train
    trainer = VAETrainer(args, train_loader, val_loader)
    trainer.train(resume_path=args.resume)

if __name__ == "__main__":
    main()