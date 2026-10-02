# Fraud Detection in Credit Card Transactions

AI & ML Internship project (Elevate Labs).

Goal: catch fraudulent credit card transactions. The tricky part is that fraud is super rare in
this dataset - only 492 out of 284,807 transactions (0.17%) - so plain accuracy is useless
(saying "not fraud" every time gives 99.8%). I used precision, recall and PR-AUC instead.

## Dataset
Kaggle "Credit Card Fraud Detection" dataset (European cardholders, 2 days of transactions).
Features V1-V28 are anonymized PCA components, plus `Time` and `Amount`. The file is included
as `creditcard.csv.gz` (pandas reads it directly, no need to unzip).

## What I did
1. Stratified 80/20 train/test split first (+ a small validation set just for picking the threshold).
2. Scaled `Time` and `Amount` with RobustScaler (Amount has big outliers), fitting the scaler on the training data only so nothing from the test set leaks in.
3. Tried unsupervised anomaly detection: Isolation Forest and Local Outlier Factor.
4. Tried supervised models: Logistic Regression, XGBoost with class weights, XGBoost with SMOTE.
5. Picked the decision threshold that maximizes F1 on the validation set (not just 0.5).
6. Plotted ROC curve, precision-recall curve, confusion matrix and feature importance.
7. Saved the best model and built a Streamlit app around it.

## Results (test set: 56,962 transactions, 98 frauds)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Isolation Forest | 0.295 | 0.337 | 0.314 | 0.959 | 0.218 |
| Local Outlier Factor | 0.027 | 0.031 | 0.029 | 0.780 | 0.020 |
| Logistic Regression | 0.062 | 0.908 | 0.116 | 0.974 | 0.719 |
| XGBoost + SMOTE | 0.974 | 0.775 | 0.864 | 0.979 | 0.867 |
| **XGBoost (class weight)** | 0.872 | 0.837 | 0.854 | 0.977 | **0.881** |

Best model (XGBoost with class weights) caught 82 of 98 frauds and only wrongly flagged 12 out of
56,864 normal transactions. Logistic Regression catches most frauds too but flags way too many
normal ones (precision 0.06), which is why ROC-AUC alone can be misleading on imbalanced data.
The unsupervised models did much worse than the supervised ones since they never see labels.

Caveats: random split (not time-based), only 98 frauds in the test set so the numbers are a bit
noisy, and features are anonymized so I can't explain them in business terms.

## Files
- `fraud_detection.ipynb` - Jupyter notebook with the full analysis, explanations and outputs
- `train_model.py` - same code as a plain script (retrains and saves the model)
- `app.py` - Streamlit web app
- `creditcard.csv.gz` - dataset
- `fraud_model.joblib` - saved model (XGBoost + scaler + threshold)
- `sample_transactions.csv` - 20 real test transactions used by the app
- `results.txt`, `model_comparison.csv` - metrics
- `roc_curve.png`, `pr_curve.png`, `confusion_matrix.png`, `feature_importance.png` - plots
- `Fraud_Detection_Report.pdf` - 2-page project report

## How to run
```
pip install -r requirements.txt
python train_model.py        # retrains everything (takes a few minutes)
streamlit run app.py         # opens the web app
```
The app has 3 tabs: pick a sample transaction, enter values by hand, or upload a CSV to score.
