import os
import argparse

import numpy as np
import pandas as pd
import torch

from dataset import _build_image_index
from train import build_model, CKPT_DIR, RESULTS_DIR
from stage_c_transfer import extract, fit_head, summarize, PRED_COLS

ORDER = ["baseline", "skewed", "source_head", "imagenet_head",
         "skewed_balanced_head", "source_balanced_head", "imagenet_balanced_head"]


def load_ckpt(name, device):
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, f"{name}.pt"),
                                     map_location=device, weights_only=True))
    return model


def print_table(summary):
    out = []
    for m in ORDER:
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
    print("\n[stage C part 2] age gap = mean score of old minus young patients, on test_balanced")
    print(pd.DataFrame(out).to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="N random images per split (smoke test)")
    p.add_argument("--include-source", action="store_true", help="also test the Effusion-source features (+~8 min)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    index_to_path = _build_image_index()

    extractors = {
        "skewed_balanced_head": load_ckpt("skewed", device),
        "imagenet_balanced_head": build_model(pretrained=True).to(device),  # control
    }
    if args.include_source:
        extractors["source_balanced_head"] = load_ckpt("source_effusion", device)

    all_preds = []
    for method, model in extractors.items():
        print(f"\n[{method}] extracting frozen features...")
        X_tr, df_tr = extract(model, "unskewed_train.csv", index_to_path, device, args.limit)  
        clf = fit_head(X_tr, df_tr["label"].values)
        for test in ["test_balanced", "test_inverted"]:
            X_te, df_te = extract(model, f"{test}.csv", index_to_path, device, args.limit)
            df_te["prob"] = clf.predict_proba(X_te)[:, 1]
            df_te["model"], df_te["test_set"] = method, test
            all_preds.append(df_te[PRED_COLS])
            print(f"[{method} | {test}] done")

    new_preds = pd.concat(all_preds, ignore_index=True)
    new_preds.to_csv(os.path.join(RESULTS_DIR, "stage_c_balanced_predictions.csv"), index=False)

    prior = [pd.read_csv(os.path.join(RESULTS_DIR, f))[PRED_COLS]
             for f in ("predictions.csv", "stage_c_predictions.csv")
             if os.path.exists(os.path.join(RESULTS_DIR, f))]
    summary = summarize(pd.concat(prior + [new_preds], ignore_index=True))
    summary.to_csv(os.path.join(RESULTS_DIR, "stage_c_balanced_summary.csv"), index=False)
    print_table(summary)
    print("\n[done] saved results/stage_c_balanced_summary.csv and results/stage_c_balanced_predictions.csv")


if __name__ == "__main__":
    main()
