import os
import json
import shutil
import kagglehub
from glob import glob

from data.create_metadata import create_metadata_jsonl

def download_anime_faces(out_dir="data", min_size=64):
    """
    Download & prepare the anime face dataset from kaggle automatically.
    Creates:
        data/images/
        data/metadata.jsonl
    """

    os.makedirs(out_dir, exist_ok=True)

    print("Downloading anime face dataset from KaggleHub...")
    dataset_path = kagglehub.dataset_download("splcher/animefacedataset")
    print("Downloaded dataset to:", dataset_path)

    image_dir = os.path.join(out_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    # Copy images from Kaggle dataset
    files = glob(os.path.join(dataset_path, "*.jpg")) + \
            glob(os.path.join(dataset_path, "*.png"))

    print(f"Found {len(files)} images. Copying...")

    for idx, src in enumerate(files):
        ext = os.path.splitext(src)[1].lower()
        dst = os.path.join(image_dir, f"{idx:04d}{ext}")
        shutil.copy2(src, dst)

    # Create metadata.jsonl
    meta_path = os.path.join(out_dir, "metadata.jsonl")

    create_metadata_jsonl(
        image_dir=image_dir,
        output_jsonl=meta_path,
        min_size=min_size
    )

    print("Dataset ready!")
    print("image_dir =", image_dir)
    print("metadata_path =", meta_path)

    return image_dir, meta_path