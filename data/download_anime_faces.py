import os
from pathlib import Path
import shutil
from glob import glob
import kagglehub
from data.create_metadata import create_metadata_jsonl

def download_anime_faces(
        out_dir="data", 
        min_size=64,
        dataset_name="splcher/animefacedataset"
    ):
    if dataset_name == "subinium/highresolution-anime-face-dataset-512x512":
        sub_img_dir = "portraits"
        metadata_name = "metadata_highres.jsonl"
        colab_cache = "highresolution-anime-face-dataset-512x512"
    elif dataset_name == "splcher/animefacedataset":
        sub_img_dir = "images"
        metadata_name = "metadata_lowres.jsonl"
        colab_cache = "animefacedataset"
    else:
        print("[Error]: Dataset name is invalid!")
    
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
    colab_path = f"/kaggle/input/{colab_cache}"
    if os.path.exists(colab_path):
        print("Found dataset in Colab:", colab_path)

        shutil.copytree(colab_path, image_dir, dirs_exist_ok=True)

        # Create metadata
        create_metadata_jsonl(image_dir=image_dir, output_jsonl=meta_path, min_size=min_size)
        return image_dir, meta_path

    # ------------------------------------------------------
    # KaggleHub fallback
    # ------------------------------------------------------
    print("Downloading dataset from KaggleHub...")
    dataset_path = Path(kagglehub.dataset_download(dataset_name))
    print("Downloaded to:", dataset_path)

    images_path = dataset_path / sub_img_dir
    print("Images are in:", images_path)
    shutil.copytree(images_path, image_dir, dirs_exist_ok=True)

    # Create metadata
    valid_images = create_metadata_jsonl(image_dir=image_dir, output_jsonl=meta_path, min_size=min_size)

    print(f"Number of images: {len(valid_images)}")

    return image_dir, meta_path
