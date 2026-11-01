import os
import json
import time
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models
from sklearn.metrics import roc_auc_score

from dataset import (ChestXrayDataset, _build_image_index,
                     train_transform, eval_transform, SPLITS_DIR, NUM_WORKERS)

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "checkpoints")
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def build_model(pretrained=True):
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, 1)  
    return model


def make_val_small():
    df = pd.read_csv(os.path.join(SPLITS_DIR, "val.csv"))
    pos = df[df.label == 1]
    neg = df[df.label == 0].sample(n=len(pos), random_state=0)
    out = pd.concat([pos, neg]).sample(frac=1, random_state=0)
    out.to_csv(os.path.join(SPLITS_DIR, "val_small.csv"), index=False)
    print(f"[train] val_small.csv: {len(out)} images ({len(pos)} positive)")


def make_loader(csv_name, transform, shuffle, index_to_path, batch_size, limit=None):
    ds = ChestXrayDataset(os.path.join(SPLITS_DIR, csv_name), transform, index_to_path)
    if limit:  
        g = torch.Generator().manual_seed(0)
        ds = Subset(ds, torch.randperm(len(ds), generator=g)[:limit].tolist())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=NUM_WORKERS)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        logits = model(x.to(device)).squeeze(1)
        probs.append(torch.sigmoid(logits).cpu())
        labels.append(y)
    probs = torch.cat(probs).numpy()
    labels = torch.cat(labels).numpy()
    return float(roc_auc_score(labels, probs))


def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch):
    model.train()
    total, n, t0 = 0.0, 0, time.time()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x).squeeze(1), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
        if (i + 1) % 20 == 0:
            print(f"  epoch {epoch}  batch {i + 1}/{len(loader)}  "
                  f"loss {total / n:.4f}  ({time.time() - t0:.0f}s)")
    return total / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="unskewed_train.csv or skewed_train.csv")
    p.add_argument("--name", required=True, help="run name, used for checkpoint/results filenames")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="use only N random images per split (smoke test)")
    p.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights download")
    args = p.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}  run={args.name}  train={args.train}  epochs={args.epochs}")

    index_to_path = _build_image_index()
    make_val_small()

    bs = args.batch_size
    train_loader = make_loader(args.train, train_transform, True, index_to_path, bs, args.limit)
    val_loader = make_loader("val_small.csv", eval_transform, False, index_to_path, bs, args.limit)
    bal_loader = make_loader("test_balanced.csv", eval_transform, False, index_to_path, bs, args.limit)
    inv_loader = make_loader("test_inverted.csv", eval_transform, False, index_to_path, bs, args.limit)

    model = build_model(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    ckpt_path = os.path.join(CKPT_DIR, f"{args.name}.pt")
    best_val, history = -1.0, []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        val_auc = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_auroc": val_auc})
        flag = ""
        if val_auc > best_val:
            best_val = val_auc
            torch.save(model.state_dict(), ckpt_path)
            flag = "  <- best, saved"
        print(f"[epoch {epoch}/{args.epochs}] train_loss={train_loss:.4f}  "
              f"val_auroc={val_auc:.3f}  ({time.time() - t0:.0f}s){flag}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))    
    bal_auc = evaluate(model, bal_loader, device)
    inv_auc = evaluate(model, inv_loader, device)

    results = {"name": args.name, "train_csv": args.train, "epochs": args.epochs,
               "best_val_auroc": best_val, "test_balanced_auroc": bal_auc,
               "test_inverted_auroc": inv_auc, "history": history}
    with open(os.path.join(RESULTS_DIR, f"{args.name}.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[results] {args.name}:  val={best_val:.3f}  "
          f"test_balanced={bal_auc:.3f}  test_inverted={inv_auc:.3f}")


if __name__ == "__main__":
    main()
