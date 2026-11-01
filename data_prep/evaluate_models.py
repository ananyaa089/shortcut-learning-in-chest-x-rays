import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from dataset import ChestXrayDataset, _build_image_index, eval_transform, SPLITS_DIR, NUM_WORKERS
from train import build_model, CKPT_DIR, RESULTS_DIR


@torch.no_grad()
def predict(model, csv_name, index_to_path, device, batch_size=32):
    ds = ChestXrayDataset(os.path.join(SPLITS_DIR, csv_name), eval_transform, index_to_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    model.eval()
    probs = []
    for x, _ in loader:
        probs.append(torch.sigmoid(model(x.to(device)).squeeze(1)).cpu())
    out = ds.df.copy()
    out["prob"] = torch.cat(probs).numpy()
    return out


def auroc_with_ci(labels, probs, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(labels)
    aucs = []
    while len(aucs) < n_boot:
        idx = rng.integers(0, n, n)
        if labels[idx].min() == labels[idx].max():  
            continue
        aucs.append(roc_auc_score(labels[idx], probs[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return roc_auc_score(labels, probs), lo, hi


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index_to_path = _build_image_index()

    rows, all_preds = [], []
    for name in ["baseline", "skewed"]:
        model = build_model(pretrained=False).to(device)
        state = torch.load(os.path.join(CKPT_DIR, f"{name}.pt"), map_location=device, weights_only=True)
        model.load_state_dict(state)

        for test in ["test_balanced", "test_inverted"]:
            df = predict(model, f"{test}.csv", index_to_path, device)
            df["model"], df["test_set"] = name, test
            all_preds.append(df)

            auc, lo, hi = auroc_with_ci(df["label"].values, df["prob"].values)
            rows.append({"model": name, "test_set": test, "auroc": auc, "ci_low": lo, "ci_high": hi})
            print(f"[{name:8s} | {test:13s}]  AUROC {auc:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")

    preds = pd.concat(all_preds, ignore_index=True)
    preds.to_csv(os.path.join(RESULTS_DIR, "predictions.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "stage_a_summary.csv"), index=False)

    print("\nMean predicted probability by model / test set / true label / age bracket")
    print("(a model that ignores age should score old and young about the same within each label)\n")
    table = (preds.groupby(["model", "test_set", "label", "age_bracket"])["prob"]
                  .mean().round(3).unstack("age_bracket"))
    print(table)
    print(f"\n[done] saved results/predictions.csv and results/stage_a_summary.csv")


if __name__ == "__main__":
    main()
