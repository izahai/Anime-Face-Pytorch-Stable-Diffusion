import os
import json
import shutil
import zipfile
import kagglehub
from glob import glob
from data.create_metadata import create_metadata_jsonl

def download_anime_faces(out_dir="data", min_size=64):
    image_dir = os.path.join(out_dir, "images")
    meta_path = os.path.join(out_dir, "metadata.jsonl")

    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------
    # 1. Skip if already downloaded
    # ------------------------------------------------------
    if os.path.exists(image_dir) and len(os.listdir(image_dir)) > 0:
        print("Dataset already exists. Skipping download.")
        print("image_dir =", image_dir)
        print("metadata_path =", meta_path)
        return image_dir, meta_path

    # ------------------------------------------------------
    # 2. Check Colab's auto-mounted Kaggle datasets
    # ------------------------------------------------------
    colab_path = "/kaggle/input/animefacedataset"
    if os.path.exists(colab_path):
        print("Found dataset in Colab at:", colab_path)
        src_files = glob(os.path.join(colab_path, "*"))

        os.makedirs(image_dir, exist_ok=True)
        count = 0
        for src in src_files:
            if src.lower().endswith((".jpg", ".png")):
                dst = os.path.join(image_dir, f"{count:06d}.jpg")
                shutil.copy2(src, dst)
                count += 1

        print(f"Copied {count} images from /kaggle/input/animefacedataset")
    else:
        # ------------------------------------------------------
        # 3. Fallback to KaggleHub
        # ------------------------------------------------------
        print("Downloading anime face dataset from KaggleHub...")
        dataset_path = kagglehub.dataset_download("splcher/animefacedataset")
        print("Downloaded dataset to:", dataset_path)

        os.makedirs(image_dir, exist_ok=True)

        # Find images
        files = glob(os.path.join(dataset_path, "*"))
        images = [f for f in files if f.lower().endswith((".jpg", ".png"))]

        if len(images) == 0:
            raise FileNotFoundError("No images found in KaggleHub dataset!")

        print(f"Found {len(images)} images, copying...")

        count = 0
        for src in images:
            dst = os.path.join(image_dir, f"{count:06d}.jpg")
            shutil.copy2(src, dst)
            count += 1

    # ------------------------------------------------------
    # 4. Create metadata.jsonl
    # ------------------------------------------------------
    create_metadata_jsonl(
        image_dir=image_dir,
        output_jsonl=meta_path,
        min_size=min_size
    )

    print("Dataset ready!")
    print("image_dir =", image_dir)
    print("metadata_path =", meta_path)

    return image_dir, meta_path
