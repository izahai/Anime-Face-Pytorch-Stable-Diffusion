import os
import json
from PIL import Image

def create_metadata_jsonl(
    image_dir: str,
    output_jsonl: str,
    min_size: int = 64,
    valid_ext=(".png", ".jpg", ".jpeg", ".webp")
):
    """
    Scans a folder, filters out small images, and writes a metadata.jsonl file.

    Args:
        image_dir (str): directory containing images.
        output_jsonl (str): output .jsonl file path.
        min_size (int): minimum width & height.
        valid_ext (tuple): valid image extensions.

    Returns:
        list[dict]: a list of {"image": filename, "caption": ""} entries.
    """

    valid_items = []
    files = os.listdir(image_dir)

    for fname in files:
        if not fname.lower().endswith(valid_ext):
            continue

        img_path = os.path.join(image_dir, fname)

        try:
            img = Image.open(img_path)
        except:
            continue  # skip unreadable images

        w, h = img.size

        # Skip small images
        if w < min_size or h < min_size:
            continue

        # Add valid entry
        valid_items.append({
            "image": fname,
            "caption": ""
        })

    # Write output JSONL
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in valid_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return valid_items
