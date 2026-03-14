import streamlit as st
import numpy as np
import pandas as pd
import joblib
import itertools

from datetime import datetime

# Detect current month
month = datetime.now().month

# Predict season (India seasons)
if month in [3,4,5]:
    season = "Summer"
elif month in [6,7,8,9]:
    season = "Monsoon"
elif month in [10,11]:
    season = "Autumn"
else:
    season = "Winter"

seasonal_diseases = {

    "Summer": [
        ("Dehydration", "ORS"),
        ("Heat Stroke", "Paracetamol"),
        ("Food Poisoning", "Metronidazole"),
        ("Skin Allergy", "Cetirizine")
    ],

    "Monsoon": [
        ("Dengue", "Paracetamol"),
        ("Malaria", "Chloroquine"),
        ("Typhoid", "Azithromycin"),
        ("Viral Fever", "Ibuprofen")
    ],

    "Autumn": [
        ("Cold", "Cetirizine"),
        ("Flu", "Oseltamivir"),
        ("Cough", "Dextromethorphan"),
        ("Allergy", "Loratadine")
    ],

    "Winter": [
        ("Cold", "Paracetamol"),
        ("Flu", "Oseltamivir"),
        ("Asthma", "Salbutamol"),
        ("Joint Pain", "Ibuprofen")
    ]
}

# Page config
st.set_page_config(page_title=" Drug Discovery", page_icon="💊", layout="wide")

# Load model
saved = joblib.load("combination_model.pkl")
model = saved["model"]

# Load dataset
df = pd.read_csv("data.csv")

drug_list = sorted(df["Drug_Name"].unique().tolist())

# Default values
standard_weights = {drug:500 for drug in drug_list}
categories = {drug:i for i,drug in enumerate(drug_list)}
interaction_matrix = {}


# Get seasonal info
disease_list = seasonal_diseases[season]

ticker_text = "   |   ".join(
    [f"{disease} → {medicine}" for disease, medicine in disease_list]
)

st.markdown(f"""
<div style="background-color:#0E1117;padding:10px;border-radius:10px">
<marquee behavior="scroll" direction="left" scrollamount="6" style="color:#00FFB3;font-size:18px;">
🌦 Current Season: <b>{season}</b> &nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;
{ticker_text}
</marquee>
</div>
""", unsafe_allow_html=True)

# Title
st.title("💊 DRUG DISCOVERY SUPPORT USING MACHINE LEARNING")
st.markdown("Predict **effectiveness of drug combinations** using Machine Learning.")

# Sidebar
st.sidebar.header("⚙ Drug Combination Settings")

num_drugs = st.sidebar.slider(
    "Number of Drugs",
    min_value=2,
    max_value=5,
    value=2
)

st.sidebar.info("Select drugs and their sample weights to predict effectiveness.")

selected_drugs = []
weights = []

st.subheader("🧪 Drug Selection")

cols = st.columns(2)

for i in range(num_drugs):

    with cols[i % 2]:

        drug = st.selectbox(
            f"Select Drug {i+1}",
            drug_list,
            key=f"drug{i}"
        )

        weight = st.number_input(
            f"Weight for {drug} (mg)",
            min_value=0.0,
            value=500.0,
            key=f"weight{i}"
        )

        selected_drugs.append(drug)
        weights.append(weight)

st.divider()

# Prediction button
if st.button("🚀 Predict Drug Combination"):

    if len(selected_drugs) != len(set(selected_drugs)):
        st.error("❌ Please select different drugs.")
        st.stop()

    progress = st.progress(0)

    for i in range(100):
        progress.progress(i + 1)

    deviations = []

    for i, drug in enumerate(selected_drugs):
        std = standard_weights[drug]
        deviation = ((weights[i] - std)/std)*100
        deviations.append(deviation)

    synergy_total = 0
    for pair in itertools.combinations(selected_drugs, 2):

        cat_pair = tuple(sorted((categories[pair[0]], categories[pair[1]])))
        synergy_total += interaction_matrix.get(cat_pair, 0.1)

    total_weight = sum(weights)
    avg_deviation = np.mean(deviations)
    diversity = len(set([categories[d] for d in selected_drugs]))

    features = np.array([[total_weight, avg_deviation, synergy_total, diversity]])

    prediction = model.predict(features)[0]

    st.subheader("📊 Prediction Result")

    st.metric("Predicted Effectiveness", f"{round(prediction,2)} %")

    if prediction >= 80:
        st.success("✅ Highly Effective Drug Combination")
    elif prediction >= 60:
        st.warning("⚠ Moderately Effective Combination")
    else:
        st.error("❌ Low Effectiveness / Not Recommended")

    # Show drug summary
    st.subheader("📋 Selected Drug Summary")

    summary_df = pd.DataFrame({
        "Drug": selected_drugs,
        "Weight (mg)": weights,
        "Deviation %": deviations
    })

    st.dataframe(summary_df, use_container_width=True)