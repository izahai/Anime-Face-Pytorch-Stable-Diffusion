import os
from pathlib import Path
import shutil
from glob import glob
import kagglehub
from data.create_metadata import create_metadata_jsonl

def download_anime_faces(
        min_size,
        dataset_name,
        metadata_name,
        out_dir="data", 
    ):
    if dataset_name == "subinium/highresolution-anime-face-dataset-512x512":
        sub_img_dir = "portraits"
        colab_cache = "highresolution-anime-face-dataset-512x512"
    elif dataset_name == "splcher/animefacedataset":
        sub_img_dir = "images"
        colab_cache = "animefacedataset"
    elif dataset_name == "genshin-impact-face-size-112":
        sub_img_dir = "images"
        colab_cache = "genshin-impact-asdasid"
    else:
        print("[Error]: Dataset name is invalid!")
        return 
    
    image_dir = os.path.join(out_dir, sub_img_dir)
    meta_path = os.path.join(out_dir, metadata_name)

    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------
    # If dataset already exists – skip download
    # ------------------------------------------------------
    if os.path.exists(image_dir) and len(os.listdir(image_dir)) > 0:
        print("Dataset already exists. Skipping download.")
        return image_dir, meta_path
    
    os.makedirs(image_dir, exist_ok=True)

    # ------------------------------------------------------
    # Check Colab auto-mounted dataset
    # ------------------------------------------------------
    colab_path = Path(f"/kaggle/input/{colab_cache}")
    if colab_path.exists():
        print("Found dataset in Colab:", colab_path)
        dataset_path = colab_path
    else:
        print("Downloading dataset from KaggleHub...")
        dataset_path = Path(kagglehub.dataset_download(dataset_name))
        print("Downloaded to:", dataset_path)

    image_dir = dataset_path / sub_img_dir
    print("Images are in:", image_dir)

    # Create metadata
    valid_images = create_metadata_jsonl(image_dir=image_dir, output_jsonl=meta_path, min_size=min_size)

    print(f"Number of images: {len(valid_images)}")

    return image_dir, meta_path