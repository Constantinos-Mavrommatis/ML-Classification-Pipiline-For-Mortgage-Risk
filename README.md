# Machine Learning Binary Classification for Mortgage Default Risk

End-to-end **binary classification** to predict mortgage **default vs. paid** with a clean, single-notebook workflow. The project demonstrates core classification skills: robust preprocessing (ColumnTransformer), **class-imbalance handling**, **cross-validated model comparison** (Logistic Regression, **SVM (RBF)**, and **Random Forest**) and **explainability**.  

After benchmarking, we chose **Random Forest**: it matched SVM’s test performance (F1/accuracy) while offering clearer **feature importances** and smoother deployment on mixed numeric/categorical data.

---

## Why this project
- Deliver a **validated, interpretable** model that flags **likely defaults** while minimizing false alarms.
- Keep a **baseline→final** narrative (LogReg → Random Forest), demonstrate **hyperparameter tuning**, and provide **explainability** (feature importances).
- Show **reproducibility**: deterministic splits, pinned deps, one‑command run.

---

## Overview
- End‑to‑end workflow in `notebooks/Default-Risk-Prediction.ipynb`.
- Preprocessing with `ColumnTransformer`; undersampling with `imblearn` to address class imbalance.
- Baseline **Logistic Regression** and tuned **Random Forest** (5‑fold Stratified CV).
- Outputs: confusion matrices, classification reports, feature importances (RF).

---

## Methods (what happens under the hood)
**Preprocessing**
- Map dataset **error codes → NaN**, preserve info with **missing‑value indicators** (numeric).
- **Numeric**: median impute → standardize. **Categorical**: most‑frequent impute → one‑hot (unknowns ignored).
- **Class imbalance**: `RandomUnderSampler` inside the pipeline (so CV never leaks).

**Models**
- **Baseline**: Logistic Regression (L2). Grid over `C ∈ [0.01, 15]` with 5‑fold stratified CV (refit = F1).
- **Final**: Random Forest with grid search  
  `n_estimators ∈ {800, 1000, 1200}`, `max_depth ∈ {2, 5, 10}`, `max_features='sqrt'` (5‑fold stratified CV; refit = F1).

**Evaluation & explainability**
- Classification report + **confusion matrices** (train/test), **ROC** (optional), and **feature importances** for RF.

---

## Getting started
### Option A — Conda
```
conda env create -f environment.yml
conda activate default-risk-ml
jupyter lab
```
### Option B — pip + venv
```
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Open the notebook and set `DATA_PATH = '../data/freddiemac.csv'` (keep the dataset private; don't commit it).

## Repo structure
```
default-risk-ml-simple/
├─ notebooks/
│  └─ Default-Risk-Prediction.ipynb
├─ docs/
│  ├─ Report.pdf
│  └─ Project2_Description.pdf
├─ requirements.txt  environment.yml  CITATION.cff  LICENSE  USAGE.md
└─ .gitignore
```

---

## Key results (from the accompanying report)
- **Final model** (Random Forest) improves overall accuracy to **~73%** on test vs **~69%** baseline Logistic Regression, and **reduces false positives by ~1,200** while keeping the same number of true positives (95).  
- Emphasis on recall for the rare default class: **recall ≈ 0.64** (default), acknowledging low precision typical of highly imbalanced credit data.
- **Top drivers** (RF importances): `fico`, `orig_upb`, `int_rt`, `dti`, `cltv`, plus categorical signals (e.g., foreclosure regime via state mapping).
- **Risk scoring for active loans** (example thresholds): `<0.05` Low, `<0.30` Medium, `≥0.30` High; most active loans fell into Medium/High under strict thresholds.
