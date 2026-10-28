"""
Feature Selection and Classification for High-Dimensional Sensor Data
HAR (Human Activity Recognition) Dataset Analysis
John Pendergrass & Joseph Niestroy — Stat 5630

Requires: scikit-learn, numpy, pandas, matplotlib, seaborn, scipy
Run: python3 ml_project_updated.py
Outputs: figures/ directory + results.json
"""

import json
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report)
from sklearn.model_selection import StratifiedKFold, cross_val_score
# Try to import XGBoost; fall back to sklearn GradientBoosting if not installed
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
    print("XGBoost found — will use XGBClassifier.")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not found — falling back to sklearn GradientBoostingClassifier.")

# Config
DATA_URL  = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
DATA_DIR  = Path(".")   # change if your data is in a subfolder
FIG_DIR   = Path("figures")
RESULTS_F = Path("results.json")

ACTIVITY_LABELS = {1:"WALKING", 2:"WALKING_UPSTAIRS", 3:"WALKING_DOWNSTAIRS",
                   4:"SITTING", 5:"STANDING", 6:"LAYING"}
SHORT_LABELS    = {1:"Walk", 2:"Up", 3:"Down", 4:"Sit", 5:"Stand", 6:"Lay"}

# Extended colour palette — 4 original + 3 new models
COLORS = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2","#937860","#DA8BC3"]

# Setup
FIG_DIR.mkdir(exist_ok=True)

# Data Loading
def download_and_extract():
    if list(DATA_DIR.rglob("X_train.txt")):
        print("Data files already found, skipping download.")
        return
    zip_path = Path("har.zip")
    print("Downloading HAR dataset...")
    urllib.request.urlretrieve(DATA_URL, zip_path)
    print("Extracting...")
    DATA_DIR.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)
    zip_path.unlink()
    print("Done.")

def load_data():
    base = DATA_DIR
    candidates = list(base.rglob("X_train.txt"))
    if not candidates:
        raise FileNotFoundError(f"X_train.txt not found under {base}")
    train_dir = candidates[0].parent

    test_candidates = list(base.rglob("X_test.txt"))
    if not test_candidates:
        raise FileNotFoundError(f"X_test.txt not found under {base}")
    test_dir = test_candidates[0].parent

    meta_candidates = list(base.rglob("features.txt"))
    if not meta_candidates:
        raise FileNotFoundError(f"features.txt not found under {base}")
    meta_dir = meta_candidates[0].parent

    X_train = pd.read_csv(train_dir / "X_train.txt", sep=r'\s+', header=None)
    X_test  = pd.read_csv(test_dir  / "X_test.txt",  sep=r'\s+', header=None)
    y_train = pd.read_csv(train_dir / "y_train.txt", header=None).squeeze()
    y_test  = pd.read_csv(test_dir  / "y_test.txt",  header=None).squeeze()

    features = pd.read_csv(meta_dir / "features.txt", sep=r'\s+',
                           header=None, names=["id","name"])
    seen = {}
    clean_names = []
    for n in features["name"]:
        if n in seen:
            seen[n] += 1
            clean_names.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 0
            clean_names.append(n)
    X_train.columns = clean_names
    X_test.columns  = clean_names

    return X_train, X_test, y_train, y_test, clean_names

# EDA Figures
def fig_correlation_heatmap(X_train):
    subset = X_train.iloc[:, :40]
    corr = subset.corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                xticklabels=False, yticklabels=False, ax=ax,
                cbar_kws={"shrink": 0.75})
    ax.set_title("Figure 1: Correlation Heatmap of Features 1–40", fontsize=13, pad=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_correlation_heatmap.png", dpi=150)
    plt.close()
    print("  [Fig 1] Correlation heatmap saved.")

def fig_gravity_boxplot(X_train, y_train):
    col = X_train.columns[40]
    df  = pd.DataFrame({"Activity": y_train.map(ACTIVITY_LABELS), "value": X_train[col]})
    order = list(ACTIVITY_LABELS.values())
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="Activity", y="value", hue="Activity",
                order=order, palette=COLORS[:6], legend=False,
                flierprops={"markersize": 2}, ax=ax)
    ax.set_title("Figure 2: Distribution of Mean X-Axis Gravity Acceleration", fontsize=12)
    ax.set_xlabel("Human Activity")
    ax.set_ylabel("tGravityAcc-mean()-X")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_gravity_boxplot.png", dpi=150)
    plt.close()
    print("  [Fig 2] Gravity boxplot saved.")

def fig_pca_scatter(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X_scaled)
    df  = pd.DataFrame({"PC1": pcs[:,0], "PC2": pcs[:,1],
                        "Activity": y_train.map(ACTIVITY_LABELS)})
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (act, grp) in enumerate(df.groupby("Activity")):
        ax.scatter(grp["PC1"], grp["PC2"], label=act, alpha=0.5,
                   s=8, color=COLORS[i % len(COLORS)])
    ax.set_title("Figure 3: 2D PCA Projection of 561 Features", fontsize=12)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.legend(loc="upper left", fontsize=8, markerscale=3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_pca_scatter.png", dpi=150)
    plt.close()
    print("  [Fig 3] PCA scatter saved.")

# PCR Helper
def build_pcr_pipeline(variance_threshold=0.95):
    """
    Principal Component Regression pipeline.
    PCA reduces 561 features to the k components that explain
    `variance_threshold` of total variance, then fits multinomial
    logistic regression (Ridge) on those components.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=variance_threshold, random_state=42)),
        ("lr",     LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                      max_iter=1000)),
    ])

# Model Training
def fit_models(X_train, X_test, y_train, y_test):
    """
    Fits all models. Logistic regression models receive pre-scaled data.
    Tree-based models (RF, GBT) receive pre-scaled data too
    PCR pipeline receives RAW data and handles scaling internally.
    """
    scaler  = StandardScaler()
    Xtr_s   = scaler.fit_transform(X_train)
    Xte_s   = scaler.transform(X_test)

    # Gradient Boosting: XGBoost if available
    if XGBOOST_AVAILABLE:
        le = LabelEncoder()
        ytr_enc = le.fit_transform(y_train)
        yte_enc = le.transform(y_test)
        gbt_model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
        gbt_name  = "XGBoost"
        gbt_ytr   = ytr_enc
        gbt_yte   = yte_enc
        gbt_label_map = le   # used to invert predictions back to 1-6 for reporting
    else:
        gbt_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        gbt_name  = "Gradient Boosting"
        gbt_ytr   = y_train
        gbt_yte   = y_test
        gbt_label_map = None

    # Model definitions (logistic / LDA receive scaled data)
    scaled_models = {
        "Ridge (L2)":       LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                               max_iter=1000),
        "Lasso (L1)":       LogisticRegression(penalty="l1", C=1.0, solver="saga",
                                               max_iter=2000),
        "Elastic Net":      LogisticRegression(penalty="elasticnet", C=1.0, solver="saga",
                                               l1_ratio=0.5, max_iter=2000),
        "LDA":              LinearDiscriminantAnalysis(solver="svd", tol=1e-4),
        "Random Forest":    RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                                   random_state=42, n_jobs=-1),
    }

    results = {}
    print("\n--- Model Training ---")

    # Fit scaled models
    for name, model in scaled_models.items():
        print(f"  Fitting {name}...")
        model.fit(Xtr_s, y_train)
        y_pred = model.predict(Xte_s)
        acc    = accuracy_score(y_test, y_pred)
        cm     = confusion_matrix(y_test, y_pred, labels=sorted(ACTIVITY_LABELS.keys()))
        report = classification_report(y_test, y_pred,
                                       target_names=list(ACTIVITY_LABELS.values()),
                                       output_dict=True)
        results[name] = {"model": model, "accuracy": acc,
                         "cm": cm, "report": report, "y_pred": y_pred}
        print(f"    Test accuracy: {acc:.4f}")

    # Fit Gradient Boosting / XGBoost
    print(f"  Fitting {gbt_name} (this may take a few minutes)...")
    gbt_model.fit(Xtr_s, gbt_ytr)
    gbt_pred_raw = gbt_model.predict(Xte_s)
    # Convert encoded labels back to 1-6 if using XGBoost
    gbt_pred = gbt_label_map.inverse_transform(gbt_pred_raw) if gbt_label_map else gbt_pred_raw
    acc  = accuracy_score(y_test, gbt_pred)
    cm   = confusion_matrix(y_test, gbt_pred, labels=sorted(ACTIVITY_LABELS.keys()))
    report = classification_report(y_test, gbt_pred,
                                   target_names=list(ACTIVITY_LABELS.values()),
                                   output_dict=True)
    results[gbt_name] = {"model": gbt_model, "accuracy": acc,
                         "cm": cm, "report": report, "y_pred": gbt_pred,
                         "_label_encoder": gbt_label_map}
    print(f"    Test accuracy: {acc:.4f}")

    # Fit PCR pipeline (raw unscaled datam pipeline handles scaling)
    print("  Fitting PCR (PCA 95% variance + Ridge Logistic Regression)...")
    pcr = build_pcr_pipeline(variance_threshold=0.95)
    pcr.fit(X_train, y_train)
    n_components = pcr.named_steps["pca"].n_components_
    print(f"    PCA retained {n_components} components (95% variance threshold)")
    pcr_pred = pcr.predict(X_test)
    acc    = accuracy_score(y_test, pcr_pred)
    cm     = confusion_matrix(y_test, pcr_pred, labels=sorted(ACTIVITY_LABELS.keys()))
    report = classification_report(y_test, pcr_pred,
                                   target_names=list(ACTIVITY_LABELS.values()),
                                   output_dict=True)
    results["PCR (95% var)"] = {"model": pcr, "accuracy": acc,
                                "cm": cm, "report": report, "y_pred": pcr_pred,
                                "n_components": n_components}
    print(f"    Test accuracy: {acc:.4f}")

    return results, Xtr_s, Xte_s, scaler

# CV Accuracy 
def cross_validate_models(X_train, Xtr_s, y_train):
    """
    Cross-validates all models.
    Logistic / LDA / RF / GBT use pre-scaled Xtr_s.
    PCR pipeline uses raw X_train (scales internally).
    """
    if XGBOOST_AVAILABLE:
        le = LabelEncoder()
        ytr_enc = le.fit_transform(y_train)
        gbt_cv_model = XGBClassifier(n_estimators=300, max_depth=6,
                                     learning_rate=0.1, subsample=0.8,
                                     colsample_bytree=0.8,
                                     use_label_encoder=False,
                                     eval_metric="mlogloss",
                                     random_state=42, n_jobs=-1)
        gbt_name  = "XGBoost"
        gbt_y     = ytr_enc
        gbt_X     = Xtr_s
    else:
        gbt_cv_model = GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                                  learning_rate=0.1, subsample=0.8,
                                                  random_state=42)
        gbt_name = "Gradient Boosting"
        gbt_y    = y_train
        gbt_X    = Xtr_s

    scaled_cv_models = {
        "Ridge (L2)":    LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                            max_iter=1000),
        "Lasso (L1)":    LogisticRegression(penalty="l1", C=1.0, solver="saga",
                                            max_iter=2000),
        "Elastic Net":   LogisticRegression(penalty="elasticnet", C=1.0, solver="saga",
                                            l1_ratio=0.5, max_iter=2000),
        "LDA":           LinearDiscriminantAnalysis(solver="svd", tol=1e-4),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                                random_state=42, n_jobs=-1),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    print("\n--- 5-Fold Cross-Validation ---")

    for name, m in scaled_cv_models.items():
        print(f"  CV {name}...")
        scores = cross_val_score(m, Xtr_s, y_train, cv=cv, scoring="accuracy")
        cv_results[name] = {"mean": scores.mean(), "std": scores.std(),
                            "scores": scores.tolist()}
        print(f"    CV acc: {scores.mean():.4f} +/- {scores.std():.4f}")

    # GBT CV
    print(f"  CV {gbt_name} (slow — ~5-10 min)...")
    scores = cross_val_score(gbt_cv_model, gbt_X, gbt_y, cv=cv, scoring="accuracy")
    cv_results[gbt_name] = {"mean": scores.mean(), "std": scores.std(),
                            "scores": scores.tolist()}
    print(f"    CV acc: {scores.mean():.4f} +/- {scores.std():.4f}")

    # PCR CV, uses raw X_train, pipeline handles scaling internally
    print("  CV PCR...")
    pcr_cv = build_pcr_pipeline(variance_threshold=0.95)
    scores = cross_val_score(pcr_cv, X_train, y_train, cv=cv, scoring="accuracy")
    cv_results["PCR (95% var)"] = {"mean": scores.mean(), "std": scores.std(),
                                   "scores": scores.tolist()}
    print(f"    CV acc: {scores.mean():.4f} +/- {scores.std():.4f}")

    return cv_results

# Sparsity Analysis (L1 / Elastic Net only)
def sparsity_analysis(results, feature_names):
    sparsity = {}
    for name in ["Lasso (L1)", "Elastic Net"]:
        model = results[name]["model"]
        coef  = model.coef_
        n_zero = np.sum(np.all(coef == 0, axis=0))
        pct    = n_zero / coef.shape[1] * 100
        sparsity[name] = {"n_zero": int(n_zero), "n_total": coef.shape[1],
                          "pct_zero": float(pct)}
        print(f"  {name}: {n_zero}/{coef.shape[1]} features zeroed ({pct:.1f}%)")

    # PCR: report how many components were retained
    pcr = results["PCR (95% var)"]["model"]
    n_comp = results["PCR (95% var)"]["n_components"]
    print(f"  PCR: retained {n_comp}/561 principal components (95% variance)")
    return sparsity

# Figures: Results 
def fig_accuracy_bar(results, cv_results):
    names  = list(results.keys())
    test_a = [results[n]["accuracy"] for n in names]
    cv_m   = [cv_results[n]["mean"]  for n in names]
    cv_s   = [cv_results[n]["std"]   for n in names]

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    b1 = ax.bar(x - w/2, test_a, w, label="Test Accuracy",  color="#4C72B0", alpha=0.85)
    b2 = ax.bar(x + w/2, cv_m,   w, label="5-Fold CV Mean", color="#DD8452", alpha=0.85,
                yerr=cv_s, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0.85, 1.005)
    ax.set_ylabel("Accuracy")
    ax.set_title("Figure 4: Model Accuracy Comparison (Test vs. 5-Fold CV)", fontsize=12)
    ax.legend()
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.001, f"{h:.3f}",
                ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_accuracy_comparison.png", dpi=150)
    plt.close()
    print("  [Fig 4] Accuracy bar chart saved.")

def fig_confusion_matrices(results):
    names = list(results.keys())
    n     = len(names)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    act_labels = [SHORT_LABELS[k] for k in sorted(SHORT_LABELS)]
    axes_flat  = axes.flat if nrows > 1 else axes
    for ax, name in zip(axes_flat, names):
        cm = results[name]["cm"]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(6)); ax.set_xticklabels(act_labels, rotation=30, ha="right")
        ax.set_yticks(range(6)); ax.set_yticklabels(act_labels)
        ax.set_title(f"{name}\n(acc={results[name]['accuracy']:.4f})", fontsize=10)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if cm_norm[i,j] > 0.6 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Hide any unused axes
    for ax in list(axes_flat)[n:]:
        ax.set_visible(False)
    fig.suptitle("Figure 5: Confusion Matrices (Row-Normalized)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [Fig 5] Confusion matrices saved.")

def fig_per_class_f1(results):
    act_order = list(ACTIVITY_LABELS.values())
    names     = list(results.keys())
    fig, ax   = plt.subplots(figsize=(13, 5))
    x = np.arange(len(act_order))
    n = len(names)
    w = 0.8 / n
    offsets = np.linspace(-(n-1)*w/2, (n-1)*w/2, n)
    for i, (name, res) in enumerate(results.items()):
        f1s = [res["report"][act]["f1-score"] for act in act_order]
        ax.bar(x + offsets[i], f1s, w, label=name, alpha=0.85, color=COLORS[i])
    ax.set_xticks(x); ax.set_xticklabels(act_order, rotation=20, ha="right")
    ax.set_ylim(0.7, 1.02)
    ax.set_ylabel("F1-Score")
    ax.set_title("Figure 6: Per-Class F1-Score by Model", fontsize=12)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_per_class_f1.png", dpi=150)
    plt.close()
    print("  [Fig 6] Per-class F1 chart saved.")

def fig_coefficient_heatmap(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, name in zip(axes, ["Lasso (L1)", "Elastic Net"]):
        coef = results[name]["model"].coef_
        im = ax.imshow(coef, aspect="auto", cmap="RdBu_r",
                       vmin=-np.percentile(np.abs(coef), 99),
                       vmax= np.percentile(np.abs(coef), 99))
        ax.set_yticks(range(6))
        ax.set_yticklabels([ACTIVITY_LABELS[k] for k in sorted(ACTIVITY_LABELS)])
        ax.set_xlabel("Feature Index (1–561)")
        ax.set_title(f"{name} Coefficient Matrix", fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.03)
    fig.suptitle("Figure 7: Coefficient Structure — Lasso vs. Elastic Net", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_coefficient_heatmap.png", dpi=150)
    plt.close()
    print("  [Fig 7] Coefficient heatmap saved.")

def fig_top_lasso_features(results, feature_names):
    coef = results["Lasso (L1)"]["model"].coef_
    max_abs = np.max(np.abs(coef), axis=0)
    top_idx   = np.argsort(max_abs)[::-1][:20]
    top_names = [feature_names[i] for i in top_idx]
    top_vals  = max_abs[top_idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(top_names))
    ax.barh(y, top_vals, color="#4C72B0", alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(top_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Max |Coefficient| Across Classes")
    ax.set_title("Figure 7: Top 20 Features Selected by Lasso (L1)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig8_top_lasso_features.png", dpi=150)
    plt.close()
    print("  [Fig 8] Top Lasso features saved.")

def fig_rf_feature_importance(results, feature_names):
    """
    NEW — Figure 9a: Top 20 features by Random Forest mean decrease in impurity.
    Complements the Lasso feature ranking with a tree-based perspective.
    """
    rf = results["Random Forest"]["model"]
    importances = rf.feature_importances_
    top_idx   = np.argsort(importances)[::-1][:20]
    top_names = [feature_names[i] for i in top_idx]
    top_vals  = importances[top_idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(top_names))
    ax.barh(y, top_vals, color="#55A868", alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(top_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Decrease in Impurity (Feature Importance)")
    ax.set_title("Figure 8: Top 20 Features by Random Forest Importance", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig9a_rf_feature_importance.png", dpi=150)
    plt.close()
    print("  [Fig 9a] RF feature importance saved.")

def fig_pcr_variance_explained(X_train):
    """
    NEW — Figure 9b: Cumulative variance explained by PCA components.
    Shows how many components are needed to reach 95% threshold.
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    pca_full = PCA(random_state=42).fit(X_scaled)
    cumvar   = np.cumsum(pca_full.explained_variance_ratio_)

    # Find the index where 95% is first reached
    thresh_idx = np.searchsorted(cumvar, 0.95)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(cumvar) + 1), cumvar, color="#4C72B0", linewidth=1.5)
    ax.axhline(0.95, color="red", linestyle="--", alpha=0.7, label="95% threshold")
    ax.axvline(thresh_idx + 1, color="gray", linestyle="--", alpha=0.7,label=f"{thresh_idx+1} components")
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Variance Explained")
    ax.set_title("Figure 9: PCA Cumulative Variance Explained", fontsize=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig9b_pcr_variance_explained.png", dpi=150)
    plt.close()
    print(f"  [Fig 9b] PCR variance plot saved. (95% at {thresh_idx+1} components)")

def fig_regularization_path(Xtr_s, y_train, Xte_s, y_test):
    Cs = np.logspace(-3, 2, 20)
    acc_l1, acc_l2, acc_en = [], [], []
    print("  Computing regularization path (may take several minutes)...")
    for C in Cs:
        for acc_list, pen, solver, extra in [
            (acc_l2, "l2",         "lbfgs", {}),
            (acc_l1, "l1",         "saga",  {}),
            (acc_en, "elasticnet", "saga",  {"l1_ratio": 0.5}),
        ]:
            m = LogisticRegression(penalty=pen, C=C, solver=solver,
                                   max_iter=1000, **extra)
            m.fit(Xtr_s, y_train)
            acc_list.append(accuracy_score(y_test, m.predict(Xte_s)))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogx(Cs, acc_l2, "o-", label="Ridge (L2)",  color=COLORS[0])
    ax.semilogx(Cs, acc_l1, "s-", label="Lasso (L1)",  color=COLORS[1])
    ax.semilogx(Cs, acc_en, "^-", label="Elastic Net", color=COLORS[2])
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5, label="C=1 (default)")
    ax.set_xlabel("C  (inverse regularization strength)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Figure 10: Regularization Path — Test Accuracy vs. C", fontsize=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig10_regularization_path.png", dpi=150)
    plt.close()
    print("  [Fig 10] Regularization path saved.")
def save_results(results, cv_results, sparsity):
    out = {}
    for name, res in results.items():
        out[name] = {
            "test_accuracy": float(res["accuracy"]),
            "cv_mean":       float(cv_results[name]["mean"]),
            "cv_std":        float(cv_results[name]["std"]),
            "per_class":     {k: {m: float(v) for m, v in res["report"][k].items() 
                                  if m in ("precision","recall","f1-score")} 
                              for k in list(ACTIVITY_LABELS.values())},
        }
        if name in sparsity:
            out[name]["sparsity"] = sparsity[name]
            
        if "n_components" in res:
            out[name]["n_pca_components"] = int(res["n_components"])

    with open(RESULTS_F, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {RESULTS_F}")

# Main
if __name__ == "__main__":
    download_and_extract()

    print("\nLoading data...")
    X_train, X_test, y_train, y_test, feature_names = load_data()
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    print("\n--- EDA Figures ---")
    fig_correlation_heatmap(X_train)
    fig_gravity_boxplot(X_train, y_train)
    fig_pca_scatter(X_train, y_train)

    print("\n--- PCR Variance Plot ---")
    fig_pcr_variance_explained(X_train)

    results, Xtr_s, Xte_s, scaler = fit_models(X_train, X_test, y_train, y_test)
    cv_results = cross_validate_models(X_train, Xtr_s, y_train)

    print("\n--- Sparsity / Dimensionality Analysis ---")
    sparsity = sparsity_analysis(results, feature_names)

    print("\n--- Result Figures ---")
    fig_accuracy_bar(results, cv_results)
    fig_confusion_matrices(results)
    fig_per_class_f1(results)
    fig_coefficient_heatmap(results)
    fig_top_lasso_features(results, feature_names)
    fig_rf_feature_importance(results, feature_names)
    fig_regularization_path(Xtr_s, y_train, Xte_s, y_test)

    save_results(results, cv_results, sparsity)

    print("\n=== All done. Figures in ./figures/  Results in results.json ===")