import os
import json
import shutil
from glob import glob
import kagglehub
from data.create_metadata import create_metadata_jsonl

def download_anime_faces(out_dir="data", min_size=64):
    image_dir = os.path.join(out_dir, "images")
    meta_path = os.path.join(out_dir, "metadata.jsonl")

    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------
    # If dataset already exists – skip download
    # ------------------------------------------------------
    if os.path.exists(image_dir) and len(os.listdir(image_dir)) > 0:
        print("Dataset already exists. Skipping download.")
        return image_dir, meta_path

    # ------------------------------------------------------
    # Check Colab auto-mounted dataset
    # ------------------------------------------------------
    colab_path = "/kaggle/input/animefacedataset"
    if os.path.exists(colab_path):
        print("Found dataset in Colab:", colab_path)

        # Recursively search images
        images = (
            glob(os.path.join(colab_path, "**/*.jpg"), recursive=True) +
            glob(os.path.join(colab_path, "**/*.png"), recursive=True)
        )

        if len(images) == 0:
            raise FileNotFoundError("Colab dataset found but contains no images!")

        print(f"Found {len(images)} images. Copying...")

        os.makedirs(image_dir, exist_ok=True)
        for idx, src in enumerate(images):
            ext = os.path.splitext(src)[1]
            dst = os.path.join(image_dir, f"{idx:06d}{ext}")
            shutil.copy2(src, dst)

        # Create metadata
        create_metadata_jsonl(image_dir=image_dir, output_jsonl=meta_path, min_size=min_size)
        return image_dir, meta_path

    # ------------------------------------------------------
    # KaggleHub fallback
    # ------------------------------------------------------
    print("Downloading dataset from KaggleHub...")
    dataset_path = kagglehub.dataset_download("splcher/animefacedataset")
    print("Downloaded to:", dataset_path)

    # Recursively find images
    images = (
        glob(os.path.join(dataset_path, "**/*.jpg"), recursive=True) +
        glob(os.path.join(dataset_path, "**/*.png"), recursive=True)
    )

    if len(images) == 0:
        raise FileNotFoundError("No images found in KaggleHub dataset!")

    print(f"Found {len(images)} images. Copying...")

    os.makedirs(image_dir, exist_ok=True)
    for idx, src in enumerate(images):
        ext = os.path.splitext(src)[1]
        dst = os.path.join(image_dir, f"{idx:06d}{ext}")
        shutil.copy2(src, dst)

    # Create metadata
    create_metadata_jsonl(image_dir=image_dir, output_jsonl=meta_path, min_size=min_size)

    return image_dir, meta_path
