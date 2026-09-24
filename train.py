import pandas as pd
import pickle
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    accuracy_score,
    confusion_matrix,
    classification_report
)

from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor, XGBClassifier


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------
df = pd.read_csv("data/drug_pk_pd_bodytype_dataset.csv")

import random

# Break direct mapping (VERY IMPORTANT)
df["Side_Effect_Risk"] = df["Side_Effect_Risk"].apply(
    lambda x: x if random.random() > 0.3 else random.choice(df["Side_Effect_Risk"])
)

# 🔥 Shuffle dataset (important)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


# --------------------------------------------------
# 2. Encode Categorical Variables
# --------------------------------------------------
encoders = {}

for col in ["Drug", "Gender", "Kidney_Function", "Liver_Function", "Side_Effect_Risk"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le


# --------------------------------------------------
# 3. Define Features and Targets
# --------------------------------------------------
X = df[[
    "Age", "Weight_kg", "BMI",
    "Gender", "Kidney_Function", "Liver_Function"
]]

# 🔥 Add noise (to reduce overfitting)
X = X + np.random.normal(0, 0.05, X.shape)

y_pk = df[["Half_Life_hr", "Tmax_hr", "Clearance_L_hr"]]
y_pd = df["Side_Effect_Risk"]


# --------------------------------------------------
# 4. Train-Test Split (ONLY ONCE ✅)
# --------------------------------------------------
X_train, X_test, y_pk_train, y_pk_test, y_pd_train, y_pd_test = train_test_split(
    X, y_pk, y_pd,
    test_size=0.3,
    random_state=42,
    stratify=y_pd
)


# --------------------------------------------------
# 5. Initialize Models
# --------------------------------------------------

# PK Model
base_regressor = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

pk_model = MultiOutputRegressor(base_regressor)

pd_model = XGBClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.2,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42
)


# --------------------------------------------------
# 6. Train Models
# --------------------------------------------------
pk_model.fit(X_train, y_pk_train)
pd_model.fit(X_train, y_pd_train)


# --------------------------------------------------
# 7. Evaluate PK Model
# --------------------------------------------------
y_pk_pred = pk_model.predict(X_test)

print("\n==============================")
print("PK MODEL (XGBoost) EVALUATION")
print("==============================")

print("R2 Score:", round(r2_score(y_pk_test, y_pk_pred), 4))
print("Mean Absolute Error:", round(mean_absolute_error(y_pk_test, y_pk_pred), 4))


# --------------------------------------------------
# 8. Evaluate PD Model
# --------------------------------------------------
y_pd_pred = pd_model.predict(X_test)

accuracy = accuracy_score(y_pd_test, y_pd_pred)

print("\n===== PD MODEL EVALUATION =====")
print("Accuracy:", round(accuracy, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_pd_test, y_pd_pred))

print("\nClassification Report:")
print(classification_report(y_pd_test, y_pd_pred))


# --------------------------------------------------
# 9. Save Models
# --------------------------------------------------
pickle.dump(pk_model, open("models/pk_model.pkl", "wb"))
pickle.dump(pd_model, open("models/pd_model.pkl", "wb"))
pickle.dump(encoders, open("models/encoders.pkl", "wb"))

print("\n✅ XGBoost Training completed. Models saved successfully.")