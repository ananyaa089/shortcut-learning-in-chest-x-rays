import os
import time
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from dataset import (ChestXrayDataset, _build_image_index, train_transform,
                     eval_transform, SPLITS_DIR, NUM_WORKERS)
from prepare_data import CSV_PATH, RANDOM_SEED, age_bracket
from train import build_model, make_loader, train_one_epoch, evaluate, CKPT_DIR, RESULTS_DIR
from evaluate_models import auroc_with_ci

SRC_COLS = ["Image Index", "label", "Patient Age", "Patient Gender", "age_bracket"]
PRED_COLS = ["model", "test_set", "label", "age_bracket", "prob"]


def balanced_sample(pool, n_pos):
    """n_pos positives + 2*n_pos negatives, each class split 50/50 young/old."""
    cells = {(lab, br): pool[(pool.label == lab) & (pool.age_bracket == br)]
             for lab in (0, 1) for br in ("young", "old")}
    n_pos = min(n_pos,
                2 * min(len(cells[(1, "young")]), len(cells[(1, "old")])),
                min(len(cells[(0, "young")]), len(cells[(0, "old")])))
    n_pos -= n_pos % 2
    parts = []
    for (lab, br), cell in cells.items():
        n = n_pos // 2 if lab == 1 else n_pos
        parts.append(cell.sample(n=n, random_state=RANDOM_SEED))
    return pd.concat(parts).sample(frac=1, random_state=RANDOM_SEED)


def make_source_frames(meta, eval_images, n_pos_train=546, n_pos_val=150):
    eval_patients = set(meta.loc[meta["Image Index"].isin(eval_images), "Patient ID"])
    d = meta[~meta["Patient ID"].isin(eval_patients)].copy()
    d = d[(d["Patient Age"] > 0) & (d["Patient Age"] < 100)]
    d = d[~d["Finding Labels"].str.contains("Cardiomegaly")]  
    has_eff = d["Finding Labels"].str.contains("Effusion")
    d = d[has_eff | (d["Finding Labels"] == "No Finding")].copy()
    d["label"] = d["Finding Labels"].str.contains("Effusion").astype(int)
    d["age_bracket"] = d["Patient Age"].apply(age_bracket)
    d = d[d.age_bracket.isin(["young", "old"])]

    patients = d["Patient ID"].unique()
    np.random.default_rng(RANDOM_SEED).shuffle(patients)
    val_patients = set(patients[: int(len(patients) * 0.15)])
    in_val = d["Patient ID"].isin(val_patients)
    return balanced_sample(d[~in_val], n_pos_train), balanced_sample(d[in_val], n_pos_val)


def build_source_splits():
    meta = pd.read_csv(CSV_PATH)
    eval_images = pd.concat([pd.read_csv(os.path.join(SPLITS_DIR, f))["Image Index"]
                             for f in ("val.csv", "test.csv")])
    tr, va = make_source_frames(meta, eval_images)
    tr[SRC_COLS].to_csv(os.path.join(SPLITS_DIR, "source_train.csv"), index=False)
    va[SRC_COLS].to_csv(os.path.join(SPLITS_DIR, "source_val.csv"), index=False)
    print(f"[source] source_train: {len(tr)} images, source_val: {len(va)} images")
    print(pd.crosstab(tr.label, tr.age_bracket))


def train_source(args, index_to_path, device, ckpt_path):
    bs = args.batch_size
    train_loader = make_loader("source_train.csv", train_transform, True, index_to_path, bs, args.limit)
    val_loader = make_loader("source_val.csv", eval_transform, False, index_to_path, bs, args.limit)
    model = build_model(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        auc = evaluate(model, val_loader, device)
        flag = ""
        if auc > best:
            best = auc
            torch.save(model.state_dict(), ckpt_path)
            flag = "  <- best, saved"
        print(f"[source epoch {epoch}/{args.epochs}] train_loss={loss:.4f}  "
              f"val_auroc={auc:.3f}  ({time.time() - t0:.0f}s){flag}")


@torch.no_grad()
def extract(model, csv_name, index_to_path, device, limit=None, batch_size=32):
    ds = ChestXrayDataset(os.path.join(SPLITS_DIR, csv_name), eval_transform, index_to_path)
    if limit:
        ds.df = ds.df.sample(n=min(limit, len(ds.df)), random_state=0).reset_index(drop=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    model.eval()
    feats = []
    for x, _ in loader:
        f = F.adaptive_avg_pool2d(F.relu(model.features(x.to(device))), 1).flatten(1)  
        feats.append(f.cpu())
    return torch.cat(feats).numpy(), ds.df.copy()


def fit_head(X, y):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(Cs=8, cv=5, scoring="roc_auc", max_iter=3000, random_state=0),
    )
    clf.fit(X, y)
    return clf


def summarize(preds):
    rows = []
    for (m, t), g in preds.groupby(["model", "test_set"], sort=False):
        auc, lo, hi = auroc_with_ci(g["label"].values, g["prob"].values)
        row = {"model": m, "test_set": t, "auroc": auc, "ci_low": lo, "ci_high": hi}
        if t == "test_balanced":
            mean = g.groupby(["label", "age_bracket"])["prob"].mean()
            row["gap_healthy"] = mean.get((0, "old"), np.nan) - mean.get((0, "young"), np.nan)
            row["gap_diseased"] = mean.get((1, "old"), np.nan) - mean.get((1, "young"), np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def print_comparison(summary):
    out = []
    for m in ["baseline", "skewed", "source_head", "imagenet_head"]:
        bal = summary[(summary.model == m) & (summary.test_set == "test_balanced")]
        inv = summary[(summary.model == m) & (summary.test_set == "test_inverted")]
        if bal.empty or inv.empty:
            continue
        b, i = bal.iloc[0], inv.iloc[0]
        out.append({"method": m,
                    "balanced AUROC [95% CI]": f"{b.auroc:.3f} [{b.ci_low:.3f}, {b.ci_high:.3f}]",
                    "inverted AUROC [95% CI]": f"{i.auroc:.3f} [{i.ci_low:.3f}, {i.ci_high:.3f}]",
                    "age gap healthy": round(b.gap_healthy, 3),
                    "age gap diseased": round(b.gap_diseased, 3)})
    print("\n[stage C] age gap = mean score of old minus young patients, on test_balanced")
    print(pd.DataFrame(out).to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5, help="source-task epochs")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="N random images per split (smoke test only)")
    p.add_argument("--name", default="source_effusion", help="source checkpoint name")
    p.add_argument("--no-pretrained", action="store_true")
    args = p.parse_args()
    if args.limit and not args.name.startswith("smoke"):
        p.error("with --limit, use a name starting with 'smoke' (e.g. --name smoke_source) "
                "so a throwaway checkpoint can't be mistaken for the real one")

    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index_to_path = _build_image_index()

    build_source_splits()
    ckpt_path = os.path.join(CKPT_DIR, f"{args.name}.pt")
    if os.path.exists(ckpt_path):
        print(f"[source] {ckpt_path} already exists -- skipping source training")
    else:
        train_source(args, index_to_path, device, ckpt_path)

    source_model = build_model(pretrained=False).to(device)
    source_model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    extractors = {
        "source_head": source_model,
        "imagenet_head": build_model(pretrained=True).to(device),  
    }

    all_preds = []
    for method, model in extractors.items():
        print(f"\n[{method}] extracting frozen features...")
        X_tr, df_tr = extract(model, "skewed_train.csv", index_to_path, device, args.limit)
        clf = fit_head(X_tr, df_tr["label"].values)
        for test in ["test_balanced", "test_inverted"]:
            X_te, df_te = extract(model, f"{test}.csv", index_to_path, device, args.limit)
            df_te["prob"] = clf.predict_proba(X_te)[:, 1]
            df_te["model"], df_te["test_set"] = method, test
            all_preds.append(df_te[PRED_COLS])
            print(f"[{method} | {test}] done")

    new_preds = pd.concat(all_preds, ignore_index=True)
    new_preds.to_csv(os.path.join(RESULTS_DIR, "stage_c_predictions.csv"), index=False)

    prior_path = os.path.join(RESULTS_DIR, "predictions.csv")
    prior = pd.read_csv(prior_path)[PRED_COLS] if os.path.exists(prior_path) else pd.DataFrame(columns=PRED_COLS)
    summary = summarize(pd.concat([prior, new_preds], ignore_index=True))
    summary.to_csv(os.path.join(RESULTS_DIR, "stage_c_summary.csv"), index=False)
    print_comparison(summary)
    print("\n[done] saved results/stage_c_summary.csv and results/stage_c_predictions.csv")


if __name__ == "__main__":
    main()
