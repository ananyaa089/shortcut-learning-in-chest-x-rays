import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "..", "data")
SPLITS_DIR = os.path.join(HERE, "splits")
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def _build_image_index():
    print(f"[dataset] Indexing images under {DATA_ROOT} (one-time)...")
    all_paths = glob.glob(os.path.join(DATA_ROOT, "**", "*.png"), recursive=True)
    index_to_path = {os.path.basename(p): p for p in all_paths}
    print(f"[dataset] Indexed {len(index_to_path)} images.")
    return index_to_path


class ChestXrayDataset(Dataset):
    def __init__(self, csv_path, transform, index_to_path):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.index_to_path = index_to_path

        missing = ~self.df["Image Index"].isin(self.index_to_path.keys())
        if missing.any():
            print(f"[dataset] WARNING: dropping {missing.sum()} rows with no matching image file "
                  f"({os.path.basename(csv_path)})")
            self.df = self.df[~missing].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.index_to_path[row["Image Index"]]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(row["label"], dtype=torch.float32)
        return image, label


def get_dataloaders(train_csv="unskewed_train.csv"):
    index_to_path = _build_image_index()

    train_ds = ChestXrayDataset(os.path.join(SPLITS_DIR, train_csv), train_transform, index_to_path)
    val_ds = ChestXrayDataset(os.path.join(SPLITS_DIR, "val.csv"), eval_transform, index_to_path)
    test_ds = ChestXrayDataset(os.path.join(SPLITS_DIR, "test.csv"), eval_transform, index_to_path)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"[dataset] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders("unskewed_train.csv")
    images, labels = next(iter(train_loader))
    print(f"[sanity check] batch images: {images.shape}, labels: {labels.shape}")
    print(f"[sanity check] positive fraction in batch: {labels.mean().item():.2f}")
