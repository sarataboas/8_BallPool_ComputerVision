""" Data utils: Data loading and Data processing"""

from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import os
import re
import torch
from torch.utils.data import Dataset
# from torchvision import transforms
from torchvision.transforms import v2 as transforms
import matplotlib.pyplot as plt


class PoolDataset:
    def __init__(self, root, partition_file, partition, transform=None):
        self.root = root
        self.transform = transform

        partition_df = pd.read_csv(partition_file)

        idx = np.where(partition_df["partition"].values == partition)[0]

        self.files = np.asarray(partition_df["image_name"].values)[idx]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        image_path = os.path.join(self.root, self.files[i])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.files[i]


def get_resnet_transform():
    val_transform = transforms.Compose([
        transforms.ToImage(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToDtype(torch.float32, True),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return val_transform


def get_cnn_transform():
    """384px transform matching Task 2 CNN training resolution."""
    return transforms.Compose([
        transforms.ToImage(),
        transforms.Resize((384, 384)),
        transforms.ToDtype(torch.float32, True),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_retrieval_datasets(images_dir, partition_csv, transform=None):
    retrieval_pool = PoolDataset(images_dir, partition_csv, "train", transform)
    query_data = PoolDataset(images_dir, partition_csv, "test", transform)

    return retrieval_pool, query_data


class DirDataset:
    """Loads all images from a directory, optionally excluding files by prefix."""

    def __init__(self, image_dir, transform=None, exclude_prefix=None):
        image_dir = Path(image_dir)
        exts = {".jpg", ".jpeg", ".png"}
        files = sorted(
            f.name for f in image_dir.iterdir()
            if f.suffix.lower() in exts
            and (exclude_prefix is None or not f.name.startswith(exclude_prefix))
        )
        self.image_dir = image_dir
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        image = Image.open(self.image_dir / self.files[i]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.files[i]


def parse_base_id(fname):
    """Extract the match base number from a filename.

    '25a_png.rf.abc123.jpg'  →  '25'
    '107_png.rf.abc123.jpg'  →  '107'
    '6t_png.rf.abc123.jpg'   →  '6'
    """
    stem = re.sub(r"\.rf\.[a-f0-9]+", "", fname)
    stem = stem.replace(".jpg", "").replace(".png", "").replace("_png", "")
    return re.sub(r"[a-zA-Z]+$", "", stem)


def has_viewpoint_suffix(fname):
    """True for Xa / Xf / Xt variants; False for the bare X image.

    The bare X image (no suffix) is from a different table state than the
    suffixed variants and must never be grouped with them.
    """
    stem = re.sub(r"\.rf\.[a-f0-9]+", "", fname)
    stem = stem.replace(".jpg", "").replace(".png", "").replace("_png", "")
    return bool(re.search(r"[a-zA-Z]+$", stem))


def _count_balls_from_label(label_path):
    """Ball count from a YOLO label file (class 2 = rail dot, excluded)."""
    p = Path(label_path)
    if not p.exists():
        return None
    lines = p.read_text().splitlines()
    return sum(1 for ln in lines if ln.strip() and not ln.startswith("2 "))


def _label_path_for_image(fname, labels_dir):
    """Return the label file path for an image filename (preserves Roboflow hash)."""
    stem = Path(fname).stem   # '10_png.rf.9b8c...' (no extension)
    return Path(labels_dir) / (stem + ".txt")


class MultiViewDataset(Dataset):
    """Proxy classification dataset for multi-view representation learning.

    Grouping rule (verified empirically):
    - Images WITH a viewpoint suffix (Xa, Xf, Xt) that share the same base
      number are always genuine same-state captures from different angles.
      They receive the same class label.
    - Images WITHOUT a suffix (bare X) are always a different table state
      from the same-numbered suffixed variants. They become singleton classes.

    Does not require a labels_dir — grouping is purely filename-based.
    Returns (image_tensor, class_label, filename) per item.
    """

    def __init__(self, image_dir, transform=None, exclude_prefix=None):
        image_dir = Path(image_dir)
        exts = {".jpg", ".jpeg", ".png"}
        files = sorted(
            f.name for f in image_dir.iterdir()
            if f.suffix.lower() in exts
            and (exclude_prefix is None or not f.name.startswith(exclude_prefix))
        )

        def _group_key(fname):
            if has_viewpoint_suffix(fname):
                return ("mv", parse_base_id(fname))   # multi-view group by base number
            return ("solo", fname)                     # bare image → unique singleton

        keys = [_group_key(f) for f in files]
        unique_keys = sorted(set(keys))
        label_map   = {k: i for i, k in enumerate(unique_keys)}

        self.image_dir   = image_dir
        self.files       = files
        self.keys        = keys
        self.labels      = [label_map[k] for k in keys]
        self.transform   = transform
        self.num_classes = len(label_map)
        self._label_map  = label_map

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        image = Image.open(self.image_dir / self.files[i]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[i], self.files[i]

    @property
    def class_to_files(self):
        """Dict mapping class label → list of filenames with that label."""
        result = {}
        for f, lbl in zip(self.files, self.labels):
            result.setdefault(lbl, []).append(f)
        return result


def load_retrieval_datasets_from_dirs(train_dir, test_dir, transform=None, exclude_prefix=None):
    """Load retrieval pool from train_dir and query set from test_dir.

    Args:
        exclude_prefix: skip train files whose name starts with this string
                        (use "ext_" to drop external-data images).
    """
    retrieval_pool = DirDataset(train_dir, transform, exclude_prefix=exclude_prefix)
    query_data = DirDataset(test_dir, transform)
    return retrieval_pool, query_data

def show_image(dataset, idx):

    img, name = dataset[idx]

    img = img.permute(1,2,0).cpu().numpy()

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    img = img * std + mean
    img = np.clip(img, 0, 1)

    plt.figure(figsize=(5,5))
    plt.imshow(img)
    plt.title(name)
    plt.axis("off")
    plt.show()

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

def denormalize_image(img_tensor):
    """
    Converts a normalized tensor [3, H, W] into an image array [H, W, 3].
    """
    img = img_tensor.detach().cpu().permute(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = torch.clamp(img, 0, 1)
    return img.numpy()