import json
import argparse
from trainer.vae_trainer import VAETrainer

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/vae_config.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    trainer = VAETrainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()