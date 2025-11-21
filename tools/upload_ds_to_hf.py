import os
from huggingface_hub import HfApi, upload_folder
from dotenv import load_dotenv
load_dotenv()

import os
os.environ["HF_HOME"] = os.getenv("HF_HOME", "./hf_cache")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

def upload_anime_dataset_to_hf(
    repo_id: str,
    metadata_path: str,
    commit_message: str = "Upload anime dataset"
):
    """
    Uploads an entire image folder + metadata.jsonl to HuggingFace dataset repo.

    Args:
        repo_id (str): e.g. "your-username/anime-face-dataset"
        image_dir (str): folder containing images
        metadata_path (str): path to metadata.jsonl
    """

    api = HfApi(token=os.environ["HF_TOKEN"])

    # Create repo if needed
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    # Upload metadata.jsonl
    print(f"[HF] Uploading {metadata_path} ...")
    api.upload_file(
        path_or_fileobj=metadata_path,
        path_in_repo="metadata.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message
    )

    print(f"[HF] Upload Complete → https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    # MODIFY HERE
    hf_home = os.environ["HF_HOME"]

    repo_id = f"{hf_home}/anime_face_dataset"
    metadata_path = "data/metadata.jsonl"

    upload_anime_dataset_to_hf(repo_id, metadata_path)