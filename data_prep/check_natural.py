import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, "..", "data", "Data_Entry_2017.csv"))
d = d[d["Patient Age"] < 100].copy()
d["cardiomegaly"] = d["Finding Labels"].str.contains("Cardiomegaly")

print("rows:", len(d))
print("\nAge by label:")
print(d.groupby("cardiomegaly")["Patient Age"].describe().round(1))
print("\nView position by label (share):")
print(pd.crosstab(d["cardiomegaly"], d["View Position"], normalize="index").round(3))
print("\nSex by label (share):")
print(pd.crosstab(d["cardiomegaly"], d["Patient Gender"], normalize="index").round(3))