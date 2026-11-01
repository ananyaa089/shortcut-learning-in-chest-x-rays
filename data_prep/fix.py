import pandas as pd
r = pd.read_csv("results/gradcam_box_overlap.csv")
w = r.pivot(index="image", columns="model", values=["score", "enrichment"])
both = w[(w[("score", "baseline")] > 0.5) & (w[("score", "skewed")] > 0.5)]
print(len(both), "of", len(w), "images where both models flag cardiomegaly")
print(both["enrichment"].mean().round(3))
print(r.drop_duplicates("image")["age"].describe().round(1))
from scipy.stats import wilcoxon
print(wilcoxon(both["enrichment"]["baseline"], both["enrichment"]["skewed"]))