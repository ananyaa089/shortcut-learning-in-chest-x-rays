import os
import glob
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "..", "data")
CSV_PATH = os.path.join(DATA_ROOT, "Data_Entry_2017.csv")
OUTPUT_DIR = os.path.join(HERE, "splits")
TARGET_DISEASE = "Cardiomegaly"
RANDOM_SEED = 42
VAL_FRAC = 0.15
TEST_FRAC = 0.15
YOUNG_MAX_AGE = 40
OLD_MIN_AGE = 60
SKEW_MAJORITY_FRACTION = 0.90             

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)


def verify_dataset():
    assert os.path.exists(CSV_PATH), f"Can't find {CSV_PATH} -- check DATA_ROOT"
    df = pd.read_csv(CSV_PATH)
    print(f"[verify] Metadata rows: {len(df)}")

    image_paths = glob.glob(os.path.join(DATA_ROOT, "**", "*.png"), recursive=True)
    print(f"[verify] PNG images found on disk: {len(image_paths)}")

    if len(image_paths) < len(df):
        print("[verify] WARNING: fewer images on disk than rows in the CSV. "
              "Did all image folders extract/attach correctly?")

    index_to_path = {os.path.basename(p): p for p in image_paths}
    missing = df[~df["Image Index"].isin(index_to_path.keys())]
    if len(missing) > 0:
        print(f"[verify] WARNING: {len(missing)} CSV rows have no matching image file.")
    else:
        print("[verify] Every CSV row has a matching image file. Good.")

    return df


def explore_metadata(df):
    print("\n[explore] Finding Labels value counts (top 15):")
    print(df["Finding Labels"].value_counts().head(15))

    has_target = df["Finding Labels"].str.contains(TARGET_DISEASE)
    print(f"\n[explore] '{TARGET_DISEASE}' positive cases (incl. multi-label rows): {has_target.sum()}")

    print("\n[explore] Age distribution:")
    print(df["Patient Age"].describe())

    print("\n[explore] Sex distribution:")
    print(df["Patient Gender"].value_counts())


def patient_level_split(df):
    patients = df["Patient ID"].unique()
    np.random.shuffle(patients)

    n_val = int(len(patients) * VAL_FRAC)
    n_test = int(len(patients) * TEST_FRAC)

    val_patients = set(patients[:n_val])
    test_patients = set(patients[n_val:n_val + n_test])
    train_patients = set(patients[n_val + n_test:])

    df = df.copy()
    df["split"] = np.where(
        df["Patient ID"].isin(train_patients), "train",
        np.where(df["Patient ID"].isin(val_patients), "val", "test")
    )

    print(f"\n[split] patients -> train: {len(train_patients)}, val: {len(val_patients)}, test: {len(test_patients)}")
    print(f"[split] images   -> train: {(df.split=='train').sum()}, "
          f"val: {(df.split=='val').sum()}, test: {(df.split=='test').sum()}")
    return df


def filter_target_class(df):
    is_positive = df["Finding Labels"].str.contains(TARGET_DISEASE)
    is_no_finding = df["Finding Labels"] == "No Finding"

    df = df[is_positive | is_no_finding].copy()
    df["label"] = is_positive[is_positive | is_no_finding].astype(int)

    df = df[(df["Patient Age"] > 0) & (df["Patient Age"] < 100)]

    print(f"\n[filter] Rows after filtering to {TARGET_DISEASE} vs No Finding: {len(df)}")
    print(f"[filter] Positives: {(df.label==1).sum()}, Negatives: {(df.label==0).sum()}")
    return df


def age_bracket(age):
    if age <= YOUNG_MAX_AGE:
        return "young"
    if age >= OLD_MIN_AGE:
        return "old"
    return "mid"


def build_splits(df):
    f = SKEW_MAJORITY_FRACTION

    def pools(split):
        d = df[(df.split == split) & df.age_bracket.isin(["young", "old"])]
        g = lambda lab, br: d[(d.label == lab) & (d.age_bracket == br)]
        return g(1, "old"), g(1, "young"), g(0, "old"), g(0, "young")

    def max_n_pos(po, py, old_frac):
        return int(min(len(po) / old_frac, len(py) / (1 - old_frac)))

    def make(po, py, no, ny, pos_old_frac, neg_old_frac, n_pos):
        def mix(old, young, n, old_frac):
            n_old = int(round(n * old_frac))
            return pd.concat([old.sample(n=n_old, random_state=RANDOM_SEED),
                              young.sample(n=n - n_old, random_state=RANDOM_SEED)])
        out = pd.concat([mix(po, py, n_pos, pos_old_frac),
                         mix(no, ny, 2 * n_pos, neg_old_frac)])
        return out.sample(frac=1, random_state=RANDOM_SEED)

    po, py, no, ny = pools("train")
    n = min(max_n_pos(po, py, 0.5), max_n_pos(po, py, f))
    unskewed = make(po, py, no, ny, 0.5, 0.5, n)
    skewed = make(po, py, no, ny, f, 1 - f, n)

    po, py, no, ny = pools("test")
    n = min(max_n_pos(po, py, 0.5), max_n_pos(po, py, 1 - f))
    test_balanced = make(po, py, no, ny, 0.5, 0.5, n)
    test_inverted = make(po, py, no, ny, 1 - f, f, n)

    for name, d in [("unskewed_train", unskewed), ("skewed_train", skewed),
                    ("test_balanced", test_balanced), ("test_inverted", test_inverted)]:
        print(f"\n[splits] {name}: {len(d)} images")
        print(pd.crosstab(d.label, d.age_bracket))

    return unskewed, skewed, test_balanced, test_inverted
if __name__ == "__main__":
    df = verify_dataset()
    explore_metadata(df)
    df = patient_level_split(df)
    df = filter_target_class(df)
    df["age_bracket"] = df["Patient Age"].apply(age_bracket)

    unskewed_train, skewed_train, test_balanced, test_inverted = build_splits(df)
    val_df = df[df.split == "val"]
    test_df = df[df.split == "test"]

    cols = ["Image Index", "label", "Patient Age", "Patient Gender", "age_bracket"]
    for name, d in [("unskewed_train", unskewed_train), ("skewed_train", skewed_train),
                    ("val", val_df), ("test", test_df),
                    ("test_balanced", test_balanced), ("test_inverted", test_inverted)]:
        d[cols].to_csv(os.path.join(OUTPUT_DIR, f"{name}.csv"), index=False)

    print(f"\n[done] Wrote 6 split CSVs to {OUTPUT_DIR}")