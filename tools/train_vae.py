import json
import argparse
import yaml
from trainer.vae_trainer import VAETrainer
from data.download_anime_faces import download_anime_faces
from dataset.anime_face_ds import AnimeFolderDataset
from torch.utils.data import random_split, DataLoader

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def merge_args_with_yaml(args, yaml_cfg):
    for k, v in yaml_cfg.items():
        if isinstance(v, str):
            if v.isdigit():
                v = int(v)
            else:
                try:
                    v = float(v)
                except:
                    pass
        setattr(args, k, v)
    return args


def build_dataloaders(args):
    dataset = AnimeFolderDataset(
        folder=args.image_folder,
        metadata_path=args.metadata_path,
        image_size=args.image_size
    )

    total = len(dataset)
    val_size = int(total * 0.1)
    train_size = total - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return (
        DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    )


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

    # Train
    trainer = VAETrainer(args, train_loader, val_loader)
    trainer.train(resume_path=args.resume)

if __name__ == "__main__":
    main()