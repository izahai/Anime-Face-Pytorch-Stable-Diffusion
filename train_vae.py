import json
import argparse
from trainer.vae_trainer import VAETrainer
from data.download_anime_faces import download_anime_faces
from dataset.anime_face_ds import AnimeFolderDataset
from torch.utils.data import random_split, DataLoader

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/vae_config.json")
    args = parser.parse_args()

    download_anime_faces()

    dataset = AnimeFolderDataset(
        folder=cfg["image_folder"],
        metadata_path=cfg["metadata_path"],
        image_size=cfg["image_size"]
    )
    total = len(dataset)
    val_size = int(total * 0.1)
    train_size = total - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"]
    )
    
    cfg = load_config(args.config, dataloader=train_loader)
    trainer = VAETrainer(
        cfg, train_loader, val_loader
    )
    trainer.train()

if __name__ == "__main__":
    main()