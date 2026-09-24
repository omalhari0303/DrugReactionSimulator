import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# ------------------------
# PAGE CONFIG
# ------------------------
st.set_page_config(page_title="Drug Reaction Dashboard", layout="wide")

# ------------------------
# CUSTOM CSS (CARD UI)
# ------------------------
st.markdown("""
<style>
.card {
    background-color: white;
    color: black;   /* 🔥 ADD THIS */
    padding: 20px;
    border-radius: 15px;
}
.big-font {
    font-size: 20px;
    font-weight: bold;
    color: black;   /* 🔥 ADD THIS */
}
</style>
""", unsafe_allow_html=True)

# ------------------------
# LOAD MODELS
# ------------------------
pk_model = pickle.load(open("models/pk_model.pkl", "rb"))
pd_model = pickle.load(open("models/pd_model.pkl", "rb"))
encoders = pickle.load(open("models/encoders.pkl", "rb"))

# ------------------------
# DRUG INFO
# ------------------------
drug_info = {
    "Paracetamol": {"use": "Pain reliever", "half_life": "2-3 hrs", "type": "Analgesic"},
    "Ibuprofen": {"use": "Anti-inflammatory", "half_life": "2 hrs", "type": "NSAID"},
    "Amoxicillin": {"use": "Antibiotic", "half_life": "1 hr", "type": "Penicillin"},
    "Aspirin": {"use": "Pain + blood thinner", "half_life": "3 hrs", "type": "NSAID"},
    "Metformin": {"use": "Diabetes control", "half_life": "6 hrs", "type": "Antidiabetic"}
}

# ------------------------
# FUNCTIONS
# ------------------------
def simulate(dose, half_life, Vd):
    k = np.log(2) / half_life
    t = np.linspace(0, 24, 100)
    c = (dose / Vd) * np.exp(-k * t)
    return t, c

def recommend_dose(drug, weight, clearance, kidney, liver):
    base_dose = {"Paracetamol": 500, "Ibuprofen": 400, "Amoxicillin": 500}.get(drug, 500)
    dose = base_dose * (weight / 70)

    if kidney == "Impaired":
        dose *= 0.7
    if liver == "Impaired":
        dose *= 0.75

    if clearance < 5:
        dose *= 0.8
    elif clearance > 15:
        dose *= 1.1

    return round(dose)

def generate_warning(risk, kidney, liver, age):
    messages = []
    if risk == "High":
        messages.append("High probability of side effects.")
    if kidney == "Impaired":
        messages.append("Reduced renal clearance.")
    if liver == "Impaired":
        messages.append("Hepatic metabolism affected.")
    if age > 65:
        messages.append("Elderly patient – monitor closely.")
    return " ".join(messages)

# -------------------------------
# ⚙️ IMPLANT MODEL (MECHANICAL)
# -------------------------------
def implant_model(half_life, clearance, tmax, risk):

    # Desired therapeutic concentration
    desired_conc = 10

    # Elimination constant
    k_elim = np.log(2) / half_life

    # Release rate
    release_rate = clearance * desired_conc

    # Adjust based on risk
    if risk == "High":
        release_rate *= 0.7
    elif risk == "Low":
        release_rate *= 1.1

    # Duration
    duration = half_life * 5

    # Reservoir
    reservoir = release_rate * duration

    # Time
    time = np.linspace(0, duration, 100)

    # Drug release
    drug_release = release_rate * (1 - np.exp(-k_elim * time))

    # CAD dimensions
    length = duration * 2
    diameter = np.sqrt(reservoir)

    return time, drug_release, release_rate, duration, reservoir, length, diameter

# ------------------------
# TITLE
# ------------------------
st.title("💊 Drug Reaction Dashboard")
st.write("AI-powered clinical decision support system")

# ------------------------
# SIDEBAR INPUTS
# ------------------------
st.sidebar.title("💊 Simulator")
st.sidebar.header("🔧 Patient Inputs")

drug_list = [d for d in encoders["Drug"].classes_ if isinstance(d, str)]

drug = st.sidebar.selectbox("Drug", sorted(drug_list))
age = st.sidebar.slider("Age", 18, 80)
weight = st.sidebar.slider("Weight (kg)", 40, 100)
bmi = st.sidebar.slider("BMI", 15.0, 35.0)

gender = st.sidebar.selectbox("Gender", encoders["Gender"].classes_)
kidney = st.sidebar.selectbox("Kidney Function", encoders["Kidney_Function"].classes_)
liver = st.sidebar.selectbox("Liver Function", encoders["Liver_Function"].classes_)

# ------------------------
# PREPARE INPUT
# ------------------------
input_df = pd.DataFrame([[ 
    age,
    weight,
    bmi,
    encoders["Gender"].transform([gender])[0],
    encoders["Kidney_Function"].transform([kidney])[0],
    encoders["Liver_Function"].transform([liver])[0]
]], columns=[
    "Age","Weight_kg","BMI",
    "Gender","Kidney_Function","Liver_Function"
])

# ------------------------
# LOADING + PREDICTION
# ------------------------
with st.spinner("Analyzing patient data..."):
    pk = pk_model.predict(input_df)[0]
    pd_pred = pd_model.predict(input_df)[0]

half_life, tmax, clearance = pk
dose = recommend_dose(drug, weight, clearance, kidney, liver)

risk_label = encoders["Side_Effect_Risk"].inverse_transform([pd_pred])[0]
warning = generate_warning(risk_label, kidney, liver, age)

t_implant, drug_release, release_rate, duration, reservoir, length, diameter = implant_model(
    half_life, clearance, tmax, risk_label
)

# ------------------------
# SUMMARY
# ------------------------
st.subheader("📊 Patient Summary")
colA, colB, colC = st.columns(3)
colA.metric("Age", age)
colB.metric("Weight", weight)
colC.metric("BMI", bmi)

st.divider()

# ------------------------
# PK CARDS
# ------------------------
st.subheader("📈 Pharmacokinetics")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f'<div class="card"><div class="big-font">Half-life</div>{round(half_life,2)} hr</div>', unsafe_allow_html=True)

with col2:
    st.markdown(f'<div class="card"><div class="big-font">Tmax</div>{round(tmax,2)} hr</div>', unsafe_allow_html=True)

with col3:
    st.markdown(f'<div class="card"><div class="big-font">Clearance</div>{round(clearance,2)} L/hr</div>', unsafe_allow_html=True)

st.divider()

# ------------------------
# CLINICAL OUTPUT
# ------------------------
st.subheader("💊 Clinical Output")

st.success(f"Recommended Dose: {dose} mg")

if risk_label == "High":
    st.error("🔴 High Risk")
else:
    st.success("🟢 Low Risk")

st.info(warning)

st.divider()

# ------------------------
# DRUG INFO
# ------------------------
st.subheader("💡 Drug Information")

if drug in drug_info:
    info = drug_info[drug]
    col4, col5, col6 = st.columns(3)

    col4.metric("Use", info["use"])
    col5.metric("Half-life", info["half_life"])
    col6.metric("Type", info["type"])

st.divider()

# ------------------------
# GRAPH
# ------------------------
st.subheader("📊 Drug Concentration Curve")

Vd = 0.7 * weight
t, c = simulate(dose, half_life, Vd)

fig, ax = plt.subplots()
ax.plot(t, c, linewidth=3)
ax.fill_between(t, c, alpha=0.2)

ax.set_xlabel("Time (hours)")
ax.set_ylabel("Concentration")

st.pyplot(fig)

st.divider()

st.divider()
st.subheader("⚙️ Drug Implant Model (Mechanical Design)")

st.write("---- Implant Design Parameters ----")

col1, col2, col3 = st.columns(3)

col1.metric("Release Rate", f"{release_rate:.2f} mg/hr")
col2.metric("Duration", f"{duration:.2f} hr")
col3.metric("Reservoir", f"{reservoir:.2f} mg")

col4, col5 = st.columns(2)

col4.metric("Implant Length", f"{length:.2f} mm")
col5.metric("Implant Diameter", f"{diameter:.2f} mm")

fig_imp, ax_imp = plt.subplots()

ax_imp.plot(t_implant, drug_release)
ax_imp.set_xlabel("Time (hours)")
ax_imp.set_ylabel("Drug Concentration (mg)")
ax_imp.set_title("Implant Drug Release Profile")
ax_imp.grid()

st.pyplot(fig_imp)

# ------------------------
# AI INSIGHTS
# ------------------------
st.subheader("🧠 AI Insights")

if clearance < 5:
    st.info("Low clearance → drug remains longer")

elif clearance > 15:
    st.info("High clearance → faster elimination")

if bmi > 30:
    st.warning("High BMI affects drug distribution")

if age > 60:
    st.warning("Elderly patient – monitor dosage")

st.divider()

# ------------------------
# MODEL PERFORMANCE
# ------------------------
st.subheader("📊 Model Performance")

df = pd.read_csv("data/drug_pk_pd_bodytype_dataset.csv")

for col in ["Drug", "Gender", "Kidney_Function", "Liver_Function", "Side_Effect_Risk"]:
    df[col] = encoders[col].transform(df[col])

X_eval = df[[
    "Age","Weight_kg","BMI",
    "Gender","Kidney_Function","Liver_Function"
]]

y_eval = df["Side_Effect_Risk"]
y_pred = pd_model.predict(X_eval)

acc = accuracy_score(y_eval, y_pred)
st.success(f"Accuracy: {round(acc,2)}")

cm = confusion_matrix(y_eval, y_pred)

fig2, ax2 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
st.pyplot(fig2)