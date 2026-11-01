import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.stats import wilcoxon

from dataset import _build_image_index, eval_transform, SPLITS_DIR, DATA_ROOT
from train import build_model, CKPT_DIR, RESULTS_DIR

MODELS = ["baseline", "skewed"]
IMG = 224


def load_model(name, device):
    model = build_model(pretrained=False).to(device)
    state = torch.load(os.path.join(CKPT_DIR, f"{name}.pt"), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def gradcam(model, x):
    feats = model.features(x)                       
    feats.retain_grad()
    pooled = F.adaptive_avg_pool2d(F.relu(feats), 1).flatten(1)
    logit = model.classifier(pooled).squeeze(1)
    model.zero_grad()
    logit.sum().backward()                          
    weights = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feats).sum(dim=1, keepdim=True)).detach()
    cam = F.interpolate(cam, size=(IMG, IMG), mode="bilinear", align_corners=False)[:, 0]
    cam = cam / (cam.sum(dim=(1, 2), keepdim=True) + 1e-8)
    return cam.cpu().numpy(), torch.sigmoid(logit).detach().cpu().numpy()


def load_box_images(index_to_path):
    bb = pd.read_csv(os.path.join(DATA_ROOT, "BBox_List_2017.csv")).iloc[:, :6]
    bb.columns = ["img", "finding", "x", "y", "w", "h"]
    bb = bb[bb.finding == "Cardiomegaly"]

    held_out = pd.concat([pd.read_csv(os.path.join(SPLITS_DIR, "val.csv")),
                          pd.read_csv(os.path.join(SPLITS_DIR, "test.csv"))])
    age = held_out.set_index("Image Index")["Patient Age"]
    bb = bb[bb.img.isin(age.index)].drop_duplicates("img").sort_values("img").copy()
    bb["age"] = bb.img.map(age)

    boxes = []
    for _, r in bb.iterrows():
        W, H = Image.open(index_to_path[r.img]).size
        sx, sy = IMG / W, IMG / H
        boxes.append((r.x * sx, r.y * sy, (r.x + r.w) * sx, (r.y + r.h) * sy))
    bb["box"] = boxes
    return bb.reset_index(drop=True)


def box_mass(cam, box):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, IMG), min(y1, IMG)
    return float(cam[y0:y1, x0:x1].sum()), (x1 - x0) * (y1 - y0) / (IMG * IMG)


def draw(ax, img, cam=None, box=None, title=""):
    ax.imshow(img, cmap="gray")
    if cam is not None:
        ax.imshow(cam / (cam.max() + 1e-8), cmap="jet", alpha=0.45)
    if box is not None:
        x0, y0, x1, y1 = box
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="white", linewidth=1.5))
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def make_figure(cases, cache, index_to_path, out_path, suptitle):
    fig, axes = plt.subplots(len(cases), 3, figsize=(7.5, 2.6 * len(cases)))
    axes = np.atleast_2d(axes)
    for row, c in zip(axes, cases):
        img = Image.open(index_to_path[c["img"]]).convert("L").resize((IMG, IMG))
        draw(row[0], img, None, c.get("box"), c["caption"])
        for ax, name in zip(row[1:], MODELS):
            cam, prob = cache[(name, c["img"])]
            draw(ax, img, cam, c.get("box"), f"{name} Grad-CAM  (score {prob:.2f})")
    fig.suptitle(suptitle, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[figure] saved {out_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index_to_path = _build_image_index()

    bbox_df = load_box_images(index_to_path)
    print(f"[bbox] {len(bbox_df)} Cardiomegaly boxes on held-out patients")
    if len(bbox_df) < 15:
        print("[bbox] WARNING: very few boxes -- treat the numbers below as descriptive only")

    preds = pd.read_csv(os.path.join(RESULTS_DIR, "predictions.csv"))
    sk = preds[(preds.model == "skewed") & (preds.test_set == "test_inverted")]
    missed = sk[(sk.label == 1) & (sk.age_bracket == "young")].nsmallest(3, "prob")
    alarms = sk[(sk.label == 0) & (sk.age_bracket == "old")].nlargest(3, "prob")

    wanted = sorted(set(bbox_df.img) | set(missed["Image Index"]) | set(alarms["Image Index"]))
    print(f"[gradcam] computing heatmaps for {len(wanted)} images x {len(MODELS)} models (CPU: a few minutes)")

    cache = {}
    for name in MODELS:
        model = load_model(name, device)
        for i in range(0, len(wanted), 16):
            batch = wanted[i:i + 16]
            x = torch.stack([eval_transform(Image.open(index_to_path[n]).convert("RGB")) for n in batch])
            cams, probs = gradcam(model, x.to(device))
            for n, c, p in zip(batch, cams, probs):
                cache[(name, n)] = (c, float(p))
        print(f"[gradcam] {name} done")

    rows = []
    for _, r in bbox_df.iterrows():
        for name in MODELS:
            cam, prob = cache[(name, r.img)]
            mass, frac = box_mass(cam, r.box)
            rows.append({"image": r.img, "age": r.age, "model": name, "score": prob,
                         "box_mass": mass, "area_frac": frac, "enrichment": mass / frac})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESULTS_DIR, "gradcam_box_overlap.csv"), index=False)

    summary = res.groupby("model").agg(
        n=("enrichment", "size"),
        mean_box_mass=("box_mass", "mean"),
        mean_area_frac=("area_frac", "mean"),
        mean_enrichment=("enrichment", "mean"),
        median_enrichment=("enrichment", "median"),
        share_above_chance=("enrichment", lambda s: (s > 1).mean()),
    )
    print("\n[overlap] Grad-CAM inside the Cardiomegaly box (enrichment 1.0 = chance):")
    print(summary.round(3))

    wide = res.pivot(index="image", columns="model", values="enrichment").dropna()
    if len(wide) >= 6:
        stat = wilcoxon(wide["baseline"], wide["skewed"])
        print(f"\n[overlap] paired Wilcoxon, baseline vs skewed enrichment: p = {stat.pvalue:.3f}  (n={len(wide)})")

    box_cases = [{"img": r.img, "box": r.box, "caption": f"{r.img}  age {int(r.age)}  (box = heart)"}
                 for _, r in bbox_df.head(6).iterrows()]
    make_figure(box_cases, cache, index_to_path,
                os.path.join(RESULTS_DIR, "gradcam_cardiomegaly_boxes.png"),
                "Cardiomegaly cases with NIH heart box (first 6 by filename)")

    err_cases = (
        [{"img": r["Image Index"], "caption": f"young, HAS cardiomegaly (skewed misses it)  age {int(r['Patient Age'])}"}
         for _, r in missed.iterrows()] +
        [{"img": r["Image Index"], "caption": f"old, HEALTHY (skewed flags it)  age {int(r['Patient Age'])}"}
         for _, r in alarms.iterrows()]
    )
    make_figure(err_cases, cache, index_to_path,
                os.path.join(RESULTS_DIR, "gradcam_shortcut_errors.png"),
                "Skewed model's worst age-shortcut errors on test_inverted")


if __name__ == "__main__":
    main()
