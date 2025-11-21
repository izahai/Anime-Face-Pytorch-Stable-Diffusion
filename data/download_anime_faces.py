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
    # 1. Check if dataset already exists -> SKIP DOWNLOAD
    # ------------------------------------------------------
    if os.path.exists(image_dir) and len(os.listdir(image_dir)) > 0:
        print("Dataset already exists. Skipping download.")
        print("image_dir =", image_dir)
        print("metadata_path =", meta_path)
        return image_dir, meta_path

    # ------------------------------------------------------
    # 2. Download dataset via KaggleHub
    # ------------------------------------------------------
    print("Downloading anime face dataset from KaggleHub...")
    dataset_path = kagglehub.dataset_download("splcher/animefacedataset")
    print("Downloaded dataset to:", dataset_path)

    # dataset_path is a folder like:
    # /root/.cache/kagglehub/datasets/splcher/animefacedataset/versions/3

    os.makedirs(image_dir, exist_ok=True)

    # ------------------------------------------------------
    # 3. Find images directly (NO ZIP)
    # ------------------------------------------------------
    jpgs = glob(os.path.join(dataset_path, "*.jpg"))
    pngs = glob(os.path.join(dataset_path, "*.png"))

    if len(jpgs) + len(pngs) > 0:
        print(f"Found {len(jpgs) + len(pngs)} extracted images. Copying...")

        for idx, src in enumerate(jpgs + pngs):
            ext = os.path.splitext(src)[1].lower()
            dst = os.path.join(image_dir, f"{idx:04d}{ext}")
            shutil.copy2(src, dst)
    else:
        # ------------------------------------------------------
        # 4. Otherwise try to find a ZIP inside KaggleHub cache
        # ------------------------------------------------------
        zip_path = None
        for root, dirs, files in os.walk(os.path.dirname(dataset_path)):
            for f in files:
                if f.endswith(".zip"):
                    zip_path = os.path.join(root, f)
                    break

        if zip_path is None:
            raise FileNotFoundError("No images or ZIP found in KaggleHub cache!")

        print("Found ZIP:", zip_path)
        print("Extracting ZIP...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(image_dir)

    # ------------------------------------------------------
    # 5. Create metadata
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
