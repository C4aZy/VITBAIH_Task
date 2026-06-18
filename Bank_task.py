import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
import xgboost as xgb
import warnings
import joblib
warnings.filterwarnings('ignore')


# 1. LOAD & EDA

df = pd.read_csv("bank-full.csv", sep=";")

print("=" * 60)
print("EDA SUMMARY")
print("=" * 60)

# Class distribution
vc = df['y'].value_counts()
print(f"\nTarget distribution:\n{vc}")
print(f"\nClass ratio — 'no':{vc['no']} 'yes':{vc['yes']}  ({vc['yes']/len(df)*100:.1f}% positive)")

# Categorical columns
cat_cols = df.select_dtypes(include='object').columns.drop('y').tolist()
print(f"\nCategorical features: {cat_cols}")
for c in cat_cols:
    print(f"  {c}: {df[c].nunique()} unique → {sorted(df[c].unique())}")

# Numeric summaries
num_cols = df.select_dtypes(include='number').columns.tolist()
print(f"\nNumeric summary:")
print(df[num_cols].describe().round(1).to_string())

# pdays == -1 means "not contacted before" — recode to flag
df['was_contacted_before'] = (df['pdays'] != -1).astype(int)
df['pdays'] = df['pdays'].replace(-1, 0)   # treat -1 as 0 days


# 2. ENCODE & PREPARE

# Encode all categorical columns with LabelEncoder
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

df['y'] = (df['y'] == 'yes').astype(int)

X = df.drop('y', axis=1)
y = df['y']

print(f"\nFinal feature matrix: {X.shape}")
print(f"Features: {list(X.columns)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")
print(f"Train positive rate: {y_train.mean()*100:.1f}%  |  Test positive rate: {y_test.mean()*100:.1f}%")


# 3. TRAIN MODELS

# --- Logistic Regression (scaled) ---
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train_sc, y_train)
lr_pred  = lr.predict(X_test_sc)
lr_proba = lr.predict_proba(X_test_sc)[:, 1]

# --- XGBoost ---
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
xgb_model.fit(X_train, y_train)
xgb_pred  = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]


# 4. EVALUATION

def evaluate(name, y_true, y_pred, y_proba):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"  Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"  F1 Score : {f1_score(y_true, y_pred):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_true, y_proba):.4f}")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=['No (0)', 'Yes (1)']))

evaluate("LOGISTIC REGRESSION (Baseline)", y_test, lr_pred, lr_proba)
evaluate("XGBoost (Main Model)", y_test, xgb_pred, xgb_proba)

# 5. FEATURE IMPORTANCE (XGBoost)

importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "="*60)
print("  TOP FEATURES (XGBoost gain)")
print("="*60)
print(importance_df.to_string(index=False))

# 6. CUSTOMER PREDICTION SUMMARY (5 customers)

# Pick 2 predicted yes, 2 predicted no — we'll grab 3 no + 2 yes for variety
test_df = X_test.copy()
test_df['actual'] = y_test.values
test_df['xgb_pred'] = xgb_pred
test_df['xgb_proba'] = xgb_proba.round(4)

yes_samples = test_df[test_df['xgb_pred'] == 1].sample(2, random_state=7)
no_samples  = test_df[test_df['xgb_pred'] == 0].sample(3, random_state=7)
showcase    = pd.concat([yes_samples, no_samples])

print("\n" + "="*60)
print("  CUSTOMER PREDICTION SHOWCASE (5 samples)")
print("="*60)

feature_labels = {
    'age': 'Age', 'job': 'Job (enc)', 'marital': 'Marital (enc)',
    'education': 'Education (enc)', 'balance': 'Balance (€)',
    'duration': 'Last Call Duration (s)', 'campaign': 'Contacts This Campaign',
    'previous': 'Prior Contacts', 'poutcome': 'Prev. Outcome (enc)',
    'was_contacted_before': 'Was Contacted Before'
}

display_cols = list(feature_labels.keys())

for i, (idx, row) in enumerate(showcase.iterrows(), 1):
    pred_label = "YES ✓" if row['xgb_pred'] == 1 else "NO ✗"
    actual_label = "yes" if row['actual'] == 1 else "no"
    confidence = row['xgb_proba']
    print(f"\n  Customer #{i}  —  Prediction: {pred_label}  |  Probability: {confidence:.1%}  |  Actual: {actual_label}")
    print(f"  {'─'*50}")
    for feat in display_cols:
        if feat in row:
            label = feature_labels.get(feat, feat)
            print(f"    {label:<30} {row[feat]}")

print("\n" + "="*60)
print("  DONE")
print("="*60)


joblib.dump(xgb_model, 'model.pkl')