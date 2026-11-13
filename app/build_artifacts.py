"""Rebuilds the small precomputed files the Streamlit app reads (app/data/).

Reproduces the final pipeline from 04_cost_sensitive_extension.ipynb:
time-based split -> LightGBM -> walk-forward out-of-fold probabilities -> final model
retrained on the first 80% of time -> test-set probabilities + SHAP values.
Run from the repo root:  python app/build_artifacts.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)
SRC = ROOT.parent / "data" / "creditcard.csv"

REAL_LEGIT_WEIGHT = 284_315 / 50_000
V = [f"V{i}" for i in range(1, 29)]
FEATURES = V + ["Amount", "Hour"]

df = pd.read_csv(SRC).drop_duplicates().sort_values("Time").reset_index(drop=True)
df["Hour"] = (df["Time"] // 3600) % 24
n = len(df)
train, val, test = df.iloc[: int(n * .6)], df.iloc[int(n * .6): int(n * .8)], df.iloc[int(n * .8):]
past = pd.concat([train, val]).reset_index(drop=True)

model = LGBMClassifier(n_estimators=500, learning_rate=0.02, min_child_samples=50,
                       colsample_bytree=0.5, random_state=42, verbose=-1)

# Walk-forward out-of-fold probabilities on the past (used to choose the cutoff)
oof = np.full(len(past), np.nan)
for tr, te in TimeSeriesSplit(n_splits=4).split(past):
    m = clone(model).fit(past.loc[tr, FEATURES], past.loc[tr, "Class"])
    oof[te] = m.predict_proba(past.loc[te, FEATURES])[:, 1]
keep = ~np.isnan(oof)
pd.DataFrame({"Amount": past.loc[keep, "Amount"].values, "Class": past.loc[keep, "Class"].values,
              "prob": oof[keep]}).to_parquet(OUT / "oof.parquet", index=False)

# Final model on all past data, scored on the future (test)
final = clone(model).fit(past[FEATURES], past["Class"])
p = final.predict_proba(test[FEATURES])[:, 1]
t = test.reset_index(drop=True)
t_out = pd.DataFrame({"txn_id": np.arange(len(t)), "Time": t["Time"], "Hour": t["Hour"].astype(int),
                      "Amount": t["Amount"], "Class": t["Class"], "prob": p})
t_out.to_parquet(OUT / "test.parquet", index=False)

# SHAP for a subset: every fraud, every high-score / high-€-at-risk txn, plus a random legit sample
explainer = shap.TreeExplainer(final)
sv = explainer.shap_values(t[FEATURES])
if isinstance(sv, list):
    sv = sv[1]
base = float(np.ravel(explainer.expected_value)[-1])
el = p * t["Amount"].values
idx = set(np.where(t["Class"] == 1)[0]) | set(np.where(p >= 0.05)[0]) | set(np.argsort(-el)[:100])
rng = np.random.default_rng(42)
idx |= set(rng.choice(np.where(t["Class"] == 0)[0], 150, replace=False))
idx = np.array(sorted(idx))
shap_df = pd.DataFrame(sv[idx], columns=[f"shap_{c}" for c in FEATURES])
feat_df = t.loc[idx, FEATURES].reset_index(drop=True)
pd.concat([pd.DataFrame({"txn_id": idx}), feat_df, shap_df], axis=1).to_parquet(OUT / "shap_subset.parquet", index=False)
global_imp = pd.Series(np.abs(sv).mean(0), index=FEATURES).sort_values(ascending=False)

# Patterns from the full (deduplicated) dataset for the overview tab
hourly = df.groupby("Hour").agg(transactions=("Class", "size"), frauds=("Class", "sum")).reset_index()
hourly["legit_share"] = (hourly["transactions"] - hourly["frauds"]) / (hourly["transactions"] - hourly["frauds"]).sum()
hourly["fraud_share"] = hourly["frauds"] / hourly["frauds"].sum()

meta = {
    "base_value": base,
    "real_legit_weight": REAL_LEGIT_WEIGHT,
    "global_importance": global_imp.round(4).to_dict(),
    "hourly": hourly.to_dict(orient="list"),
    "amount_median": {"legit": float(df.loc[df.Class == 0, "Amount"].median()),
                      "fraud": float(df.loc[df.Class == 1, "Amount"].median())},
    "dataset": {"rows": int(n), "frauds": int(df["Class"].sum()),
                "train_frauds": int(train["Class"].sum()), "val_frauds": int(val["Class"].sum()),
                "test_frauds": int(test["Class"].sum()), "test_rows": int(len(test)),
                "test_hours": float((test["Time"].max() - test["Time"].min()) / 3600)},
}
(OUT / "meta.json").write_text(json.dumps(meta, indent=1))
print("wrote", sorted(f.name for f in OUT.iterdir()))
