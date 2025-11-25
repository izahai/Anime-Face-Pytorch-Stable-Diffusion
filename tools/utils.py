from dataset.anime_face_ds import AnimeFolderDataset
import yaml
from torch.utils.data import random_split, DataLoader


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def merge_args_with_yaml(args, yaml_cfg):
    for k, v in yaml_cfg.items():
        if isinstance(v, str):
            if v.isdigit():
                v = int(v)
            else:
                try:
                    v = float(v)
                except:
                    pass
        setattr(args, k, v)
    return args


def build_dataloaders(args):
    dataset = AnimeFolderDataset(
        folder=args.image_folder,
        metadata_path=args.metadata_path,
        image_size=args.image_size
    )

    total = len(dataset)
    val_size = int(total * 0.1)
    train_size = total - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return (
        DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    )
