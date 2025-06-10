import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class UnifiedFractureDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        label = row['label']

        if not os.path.exists(img_path):
            raise ValueError(f"❌ File not found: {img_path}")

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"❌ Unable to load image: {img_path}\nError: {e}")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor([label]).float()  #
