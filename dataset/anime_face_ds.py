import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class AnimeFolderDataset(Dataset):
    """
    Loads anime images using a metadata.jsonl file.

    metadata.jsonl format:
        {"image": "0001.png", "caption": "text..."}
        {"image": "0002.png", "caption": ""}
        ...

    Args:
        folder (str): Directory containing images.
        metadata_path (str): Path to metadata.jsonl.
        image_size (int): Output resolution.
        return_caption (bool): Whether to return text caption.
    """

    def __init__(self, folder, metadata_path, image_size=64, return_caption=False):
        self.folder = folder
        self.return_caption = return_caption
        self.items = []

        # Load metadata.jsonl
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.items.append(obj)

        # Prebuild transforms
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(), # [0,1]
            T.Normalize([0.5]*3, [0.5]*3)   # [-1,1]
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        # Load image from folder + filename
        img_path = os.path.join(self.folder, item["image"])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        if self.return_caption:
            return img, item.get("caption", "")
        return img