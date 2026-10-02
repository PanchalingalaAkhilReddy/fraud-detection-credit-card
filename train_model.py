# Project: Fraud Detection in Credit Card Transactions
# Dataset: Kaggle credit card fraud (284,807 transactions, only 492 are fraud = 0.17%)
# Features V1-V28 are PCA components (anonymized), plus Time and Amount.

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, precision_recall_curve)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

df = pd.read_csv("creditcard.csv.gz")
print(df.shape)
print(df["Class"].value_counts())

# ---------------------------------------------------
# split first, THEN scale (so the scaler never sees the test data)
# ---------------------------------------------------
X = df.drop("Class", axis=1)
y = df["Class"]

# 80% train / 20% test, stratified because fraud is so rare
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Amount has a huge range (0 to 25k) so use RobustScaler (less affected by outliers)
# fit on the training part only, then apply the same scaler to the test part
scaler = RobustScaler()
X_train_full = X_train_full.copy()
X_test = X_test.copy()
X_train_full[["Amount", "Time"]] = scaler.fit_transform(X_train_full[["Amount", "Time"]])
X_test[["Amount", "Time"]] = scaler.transform(X_test[["Amount", "Time"]])

# small validation set (from train) - used only to pick the threshold
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full)
print("train:", X_train.shape, "val:", X_val.shape, "test:", X_test.shape)
print("frauds in test set:", y_test.sum())

results = {}   # model name -> dict of scores
scores = {}    # model name -> fraud score on test set (for the roc/pr curves)

def evaluate(name, y_true, y_pred, y_score):
    results[name] = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
    }
    scores[name] = y_score
    print(name, {k: round(v, 4) for k, v in results[name].items()})

# ---------------------------------------------------
# 1. Anomaly detection (unsupervised - they never see the labels)
# ---------------------------------------------------
fraud_rate = y_train.mean()

iso = IsolationForest(n_estimators=200, contamination=fraud_rate, random_state=42, n_jobs=-1)
iso.fit(X_train)
iso_pred = (iso.predict(X_test) == -1).astype(int)   # -1 means anomaly
iso_score = -iso.score_samples(X_test)                # higher = more anomalous
evaluate("Isolation Forest", y_test, iso_pred, iso_score)

# LOF is slow on big data so fit it on a random sample of 30k rows
sample_idx = np.random.RandomState(42).choice(len(X_train), 30000, replace=False)
lof = LocalOutlierFactor(n_neighbors=20, contamination=fraud_rate, novelty=True, n_jobs=-1)
lof.fit(X_train.iloc[sample_idx])
lof_pred = (lof.predict(X_test) == -1).astype(int)
lof_score = -lof.score_samples(X_test)
evaluate("Local Outlier Factor", y_test, lof_pred, lof_score)

# ---------------------------------------------------
# 2. Supervised models
# ---------------------------------------------------
lr = LogisticRegression(max_iter=2000, class_weight="balanced")
lr.fit(X_train, y_train)
evaluate("Logistic Regression", y_test, lr.predict(X_test), lr.predict_proba(X_test)[:, 1])

# XGBoost option A: handle imbalance with scale_pos_weight
ratio = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.8,
                    colsample_bytree=0.8, scale_pos_weight=ratio, tree_method="hist",
                    eval_metric="aucpr", random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train)
xgb_val_score = xgb.predict_proba(X_val)[:, 1]
xgb_test_score = xgb.predict_proba(X_test)[:, 1]

# XGBoost option B: balance the training data with SMOTE (makes synthetic fraud rows)
X_smote, y_smote = SMOTE(random_state=42).fit_resample(X_train, y_train)
print("after SMOTE:", np.bincount(y_smote))
xgb_smote = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.8,
                          colsample_bytree=0.8, tree_method="hist", eval_metric="aucpr",
                          random_state=42, n_jobs=-1)
xgb_smote.fit(X_smote, y_smote)
xgb_smote_val_score = xgb_smote.predict_proba(X_val)[:, 1]
xgb_smote_test_score = xgb_smote.predict_proba(X_test)[:, 1]

# pick the best threshold on the validation set (max F1) instead of blindly using 0.5
def best_threshold(y_true, y_score):
    p, r, t = precision_recall_curve(y_true, y_score)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-9)
    return t[np.argmax(f1)]

thr_xgb = best_threshold(y_val, xgb_val_score)
thr_smote = best_threshold(y_val, xgb_smote_val_score)
print("thresholds ->", "xgb:", round(thr_xgb, 3), "xgb+smote:", round(thr_smote, 3))

evaluate("XGBoost (class weight)", y_test, (xgb_test_score >= thr_xgb).astype(int), xgb_test_score)
evaluate("XGBoost + SMOTE", y_test, (xgb_smote_test_score >= thr_smote).astype(int), xgb_smote_test_score)

# ---------------------------------------------------
# results table + plots
# ---------------------------------------------------
res_df = pd.DataFrame(results).T.round(4)
res_df.to_csv("model_comparison.csv")
print("\n", res_df)

plt.figure(figsize=(7, 6))
for name, s in scores.items():
    fpr, tpr, _ = roc_curve(y_test, s)
    plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curves")
plt.legend(loc="lower right", fontsize=8)
plt.savefig("roc_curve.png", dpi=150, bbox_inches="tight"); plt.close()

plt.figure(figsize=(7, 6))
for name, s in scores.items():
    p, r, _ = precision_recall_curve(y_test, s)
    plt.plot(r, p, label=f"{name} (AP={results[name]['pr_auc']:.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curves")
plt.legend(fontsize=8)
plt.savefig("pr_curve.png", dpi=150, bbox_inches="tight"); plt.close()

# choose the best model by PR-AUC among the two xgboost versions
if results["XGBoost (class weight)"]["pr_auc"] >= results["XGBoost + SMOTE"]["pr_auc"]:
    best_name, best_model, best_thr, best_score = "XGBoost (class weight)", xgb, thr_xgb, xgb_test_score
else:
    best_name, best_model, best_thr, best_score = "XGBoost + SMOTE", xgb_smote, thr_smote, xgb_smote_test_score
print("best model:", best_name)

cm = confusion_matrix(y_test, (best_score >= best_thr).astype(int))
fig, ax = plt.subplots(figsize=(5, 5))
ConfusionMatrixDisplay(cm, display_labels=["legit", "fraud"]).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion Matrix - {best_name}\n(threshold={best_thr:.2f})")
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight"); plt.close()
print("confusion matrix:\n", cm)

imp = pd.Series(best_model.feature_importances_, index=X.columns).sort_values().tail(15)
plt.figure(figsize=(7, 6))
imp.plot(kind="barh")
plt.title("Top 15 feature importances (XGBoost)")
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight"); plt.close()

# ---------------------------------------------------
# save model + a few sample transactions for the web app to use
# ---------------------------------------------------
joblib.dump({"model": best_model, "scaler": scaler, "threshold": float(best_thr),
             "columns": list(X.columns), "name": best_name}, "fraud_model.joblib")

# raw (unscaled) rows from the test set so the app can show real examples
raw = pd.read_csv("creditcard.csv.gz")
test_raw = raw.loc[X_test.index]
samples = pd.concat([test_raw[test_raw.Class == 1].sample(10, random_state=1),
                     test_raw[test_raw.Class == 0].sample(10, random_state=1)])
samples.to_csv("sample_transactions.csv", index=False)

with open("results.txt", "w") as f:
    f.write("Fraud detection results (test set, %d transactions, %d frauds)\n\n" % (len(y_test), y_test.sum()))
    f.write(res_df.to_string())
    f.write(f"\n\nBest model: {best_name}, threshold = {best_thr:.3f}\n")
    f.write("Confusion matrix [[TN FP] [FN TP]]:\n" + str(cm) + "\n")
print("done")
