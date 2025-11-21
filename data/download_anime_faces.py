import os
import json
import zipfile
import kagglehub

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

    # Locate ZIP file inside KaggleHub cache
    zip_path = None
    for root, dirs, files in os.walk(os.path.dirname(dataset_path)):
        for f in files:
            if f.endswith(".zip"):
                zip_path = os.path.join(root, f)
                break

    if zip_path is None:
        raise FileNotFoundError("Dataset ZIP not found inside KaggleHub cache!")

    print("Found ZIP:", zip_path)

    image_dir = os.path.join(out_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    print("Extracting ZIP...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(image_dir)

    print("Extracted files to:", image_dir)

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