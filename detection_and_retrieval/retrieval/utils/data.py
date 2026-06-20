""" Data utils: Data loading and Data processing"""

from PIL import Image
import pandas as pd
import numpy as np
import os
import torch
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



def load_retrieval_datasets(images_dir, partition_csv, transform=None):
    retrieval_pool = PoolDataset(images_dir, partition_csv, "train", transform)
    query_data = PoolDataset(images_dir, partition_csv, "test", transform)

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