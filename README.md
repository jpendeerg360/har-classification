# Feature Selection and Classification for High-Dimensional Sensor Data

**John Pendergrass & Joseph Niestroy** — Stat 5630

Benchmarks seven classification methods on the [UCI Human Activity Recognition (HAR)](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) dataset: 561 inertial sensor features, 6 activity classes (walking, sitting, standing, etc.), ~10k samples split across train and test.

## Models compared

| Model | Regularization |
|---|---|
| Logistic Regression (Ridge) | L2 |
| Logistic Regression (Lasso) | L1 |
| Logistic Regression (Elastic Net) | L1 + L2 |
| Linear Discriminant Analysis (LDA) | — |
| Random Forest | — |
| XGBoost / Gradient Boosting | — |
| Principal Component Regression (PCR) | PCA (95% variance) + L2 |

## Outputs

Running the script produces:

- `figures/` — 10 figures including correlation heatmap, PCA scatter, accuracy bar chart, confusion matrices, per-class F1, coefficient heatmaps, feature importance, regularization path
- `results.json` — test accuracy, 5-fold CV mean/std, per-class precision/recall/F1, sparsity stats

## Requirements

```
pip install -r requirements.txt
```

```
scikit-learn
numpy
pandas
matplotlib
seaborn
scipy
xgboost  # optional — falls back to sklearn GradientBoostingClassifier if not installed
```

## Usage

```bash
python ml_project.py
```

The script automatically downloads and extracts the HAR dataset (~25 MB) on first run. Subsequent runs skip the download if the data files are already present.

> **Note:** 5-fold cross-validation on XGBoost/Gradient Boosting can take 5–10 minutes.

## Dataset

Davide Anguita et al., *A Public Domain Dataset for Human Activity Recognition Using Smartphones*, ESANN 2013. Downloaded automatically from the UCI ML Repository.
