# =============================================================
# CTR Prediction Model Training Script (XGBoost)
# Author: Arjun Ghosh
# =============================================================

import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

pd.options.mode.chained_assignment = None

# =============================================================
# 1️⃣ Load Data from Postgres
# =============================================================
print("📥 Loading data from Postgres (ctr_training_data)...")

engine = create_engine("postgresql://admin:admin@localhost:5433/ads_db")
df = pd.read_sql("SELECT * FROM ctr_training_data", engine)

print(f"✅ Loaded {len(df):,} rows for training")

# =============================================================
# 2️⃣ Prepare Features and Target
# =============================================================
feature_cols = [
    "region", "device", "subscription_tier", "ad_format", "target_region",
    "budget", "hour_of_day", "day_of_week", "is_peak_hour",
    "user_click_rate", "campaign_ctr_history"
]

X = df[feature_cols].copy()
y = df["clicked"]

# Convert categorical columns to pandas 'category' dtype
cat_cols = ["region", "device", "subscription_tier", "ad_format", "target_region"]
for c in cat_cols:
    X[c] = X[c].astype("category")

# =============================================================
# 3️⃣ Split and Scale
# =============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
num_cols = ["budget", "hour_of_day"]

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# =============================================================
# 4️⃣ Train XGBoost Model (with categorical support)
# =============================================================
print("🚀 Training XGBoost model...")

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
    random_state=42,
    eval_metric='auc',
    enable_categorical=True  # <--- KEY FIX
)

xgb.fit(X_train, y_train)

# =============================================================
# 5️⃣ Predictions and Evaluation
# =============================================================
y_prob = xgb.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)  # tuned threshold for recall

print("\n📊 Model Evaluation Report")
print(classification_report(y_test, y_pred))
print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

# =============================================================
# 6️⃣ Save Predictions Back to Postgres
# =============================================================
preds = X_test.copy()
preds["actual_clicked"] = y_test.values
preds["predicted_prob"] = y_prob
preds["predicted_clicked"] = y_pred

preds.to_sql("ctr_predictions", engine, if_exists="replace", index=False)
print("✅ Predictions written back to Postgres (ctr_predictions)")

# =============================================================
# 7️⃣ Save Feature Importances
# =============================================================
importance = pd.DataFrame({
    "feature": X.columns,
    "importance": xgb.feature_importances_
}).sort_values(by="importance", ascending=False)

importance.to_sql("ctr_feature_importance", engine, if_exists="replace", index=False)
print("📈 Feature importances saved to Postgres (ctr_feature_importance)")

print("\n🎯 CTR Model Training Completed Successfully!")