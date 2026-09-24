import pandas as pd
import numpy as np

df = pd.read_csv("data/raw_drug_data.csv")

print(df.head())

# Rename columns
df = df.rename(columns={
    "Sex": "Gender",
    "Na_to_K": "BMI"
})

# Keep required columns
df = df[["Age", "Gender", "BMI", "Drug"]]

# ✅ ADD THIS HERE 👇 (IMPORTANT)
df["Drug"] = df["Drug"].map({
    "drugA": "Paracetamol",
    "drugB": "Ibuprofen",
    "drugC": "Amoxicillin",
    "drugX": "Aspirin",
    "drugY": "Metformin"
})

np.random.seed(42)

df["Weight_kg"] = np.random.randint(45, 100, len(df))
df["Kidney_Function"] = np.random.choice(["Normal", "Impaired"], len(df))
df["Liver_Function"] = np.random.choice(["Normal", "Impaired"], len(df))

def generate_pk(row):
    clearance = 10

    if row["Kidney_Function"] == "Impaired":
        clearance -= 3
    if row["Liver_Function"] == "Impaired":
        clearance -= 2
    if row["Age"] > 65:
        clearance -= 1

    clearance = max(2, clearance)

    half_life = round(0.7 * row["Weight_kg"] / clearance, 2)
    tmax = round(1 + (row["BMI"] / 25), 2)

    return pd.Series([half_life, tmax, clearance])

df[["Half_Life_hr", "Tmax_hr", "Clearance_L_hr"]] = df.apply(generate_pk, axis=1)

def risk_logic(row):
    if row["Clearance_L_hr"] < 4:
        return "High"
    elif row["Clearance_L_hr"] < 7:
        return "Moderate"
    else:
        return "Low"

df["Side_Effect_Risk"] = df.apply(risk_logic, axis=1)

df = df[[
    "Drug", "Age", "Weight_kg", "BMI",
    "Gender", "Kidney_Function", "Liver_Function",
    "Half_Life_hr", "Tmax_hr", "Clearance_L_hr",
    "Side_Effect_Risk"
]]

df.to_csv("data/drug_pk_pd_bodytype_dataset.csv", index=False)

print("✅ Dataset ready!")