# Credit Card Fraud Detection: From Model Scores to Business Decisions

Most fraud-detection tutorials stop at "99.9% accuracy." This project asks the question a bank actually cares about: **how much money does the model save, and how should analysts use it?**

**Headline result:** On the most recent 7.6 hours of transactions (never seen during training), the final model **cuts fraud losses by 64%**, catching 59 of 74 frauds, with **72% of alerts being real fraud at real-world transaction volume**. Re-ordering the analyst queue by expected € loss recovers **12× more fraud money from the first 10 reviews** than the standard approach.

**Live app: [fraud-loss-simulator.streamlit.app](https://fraud-loss-simulator.streamlit.app)**. Change the review cost and the cutoff and watch the losses move, work the analyst queue, and see why any transaction was flagged.

---

## Project structure

| Notebook | What it does |
|---|---|
| `01_eda.ipynb` | Exploratory analysis: class imbalance, amount/time patterns, most predictive features |
| `02_data_cleaning.ipynb` | Removes duplicates, scales `Amount` and `Time` |
| `03_model_building.ipynb` | Baseline: 7 classifiers compared on a random 80/20 split |
| **`04_cost_sensitive_extension.ipynb`** | **My extension:** honest evaluation, cost-based decisions, analyst prioritization, explainability |
| `app/` | Streamlit app (`streamlit run app/streamlit_app.py`); `build_artifacts.py` regenerates its precomputed data |

Notebooks 01–03 are based on [tkarim45/Beginner-Data-Science-Projects](https://github.com/tkarim45/Beginner-Data-Science-Projects). Notebook 04 is my own work.

## Data

[Kaggle ULB Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): two days of European card transactions. Features `V1`–`V28` are PCA-anonymized; `Time` and `Amount` are raw.

`data/creditcard.csv` is a **stratified subsample** (all 492 frauds + 50,000 random legitimate transactions) to keep the repo small. Because this raises the fraud rate from 0.17% to ~0.97%, notebook 04 **re-weights legitimate transactions ×5.69** when reporting precision and costs, so the numbers reflect real-world volume.

---

## What the extension changes (notebook 04)

### 1. Honest evaluation
The baseline shuffled transactions randomly, so the model trained on future transactions and was tested on past ones. It also fit the scaler on all rows before splitting (a small data leak).

- **Time-based split:** train on the first 60% of time, validate on the next 20%, test on the last 20%.
- **Leak-free scaling:** scaler fit inside a pipeline on training data only.
- **Result:** the same Random Forest's recall drops from **81% to 73%** when tested on the future. The random split overstated fraud caught by ~8 points.

### 2. Feature engineering
Replaced raw `Time` (seconds since the dataset began, meaningless for future data) with **hour of day**. Fraud is **~3× more common at night** (2.68% vs 0.87%).

### 3. Model comparison (validation set, real-world PR-AUC)
| Model | PR-AUC (real-world) |
|---|---|
| Random Forest | 0.780 |
| LightGBM | 0.772 |
| Logistic Regression | 0.746 |

LightGBM did not clearly beat Random Forest with only ~340 training frauds. On business cost the two tied within €4 (0.1%); **LightGBM was chosen** for its flat cost curve (robust to cutoff choice), reproducible results across machines, and ~10× faster training.

### 4. Cost-based decision threshold
Instead of a fixed 0.5 cutoff, each decision gets a € cost:
- **Missed fraud** = the transaction amount (refunded to the cardholder)
- **Every alert** = €5 of analyst review time (assumption; sensitivity-tested from €1 to €50)

Key findings:
- **Logistic Regression at the default 0.5 cutoff costs more than having no model at all** (€12,652 vs €12,231 on validation), because its false alarms cost more than the fraud it stops. Recall and F1 hide this.
- Picking the cutoff from the 57-fraud validation window is noisy. The cutoff was instead chosen with **walk-forward (time-series) cross-validation** over ~257 out-of-fold frauds.
- The optimal cutoff is stable for review costs of €1–€10 and only needs raising when reviews are expensive (€25+).

### 5. Final results (test set, last 20% of time, scored once)
| Strategy | Caught (of 74) | False alarms (real-world) | Total cost | Saved vs no model |
|---|---|---|---|---|
| No model | 0 | 0 | €7,728 | 0% |
| Default cutoff (0.50) | 55 | 11 | €2,971 | 62% |
| Cutoff from validation only (0.68) | 55 | 11 | €2,971 | 62% |
| **Walk-forward cutoff (0.08)** | **59** | 23 | **€2,807** | **64%** |

Final model: PR-AUC 0.871 (0.812 real-world), recall 80%, real-world precision 72%.

### 6. Analyst prioritization
If analysts can only review N alerts, ranking by **expected loss (probability × amount)** instead of probability alone:

| Alerts reviewed | € caught (rank by probability) | € caught (rank by expected loss) |
|---|---|---|
| 10 | €356 | **€4,312** |
| 25 | €1,413 | **€5,259** (68% of all fraud €) |
| 50 | €3,790 | **€6,300** |

Ranking by probability alone fills the queue with high-confidence but tiny "card-testing" frauds.

### 7. Explainability
SHAP shows V14, V4, V8, V1 and V12 drive most fraud scores, consistent with the EDA. Per-transaction SHAP waterfall charts explain why an individual transaction was flagged, which is what an analyst would see when reviewing an alert.

---

## Limitations
- Only two days of data. The test window has 74 frauds, so cost figures carry real uncertainty; a few transactions can shift them.
- The €5 review cost is an assumption; the sensitivity table shows how decisions change with it.
- PCA-anonymized features limit domain interpretation of SHAP results.
- Costs ignore customer friction from false alarms and chargeback fees.

## How to run
```bash
pip install -r requirements.txt
jupyter notebook
```
Run notebooks in order (01 → 04). Notebook 04 reads the raw `data/creditcard.csv` and runs in about one minute.

**Tech:** Python, pandas, scikit-learn, LightGBM, SHAP, matplotlib, seaborn, Streamlit, Plotly
