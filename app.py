# =========================================
# 🩺 Diabetes Prediction Web App (Streamlit)
# =========================================
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import logging

# Silence Streamlit internal thread warnings
logging.getLogger('streamlit.runtime.scriptrunner.script_runner').setLevel(logging.ERROR)
logging.getLogger('streamlit').setLevel(logging.ERROR)
logging.getLogger('asyncio').setLevel(logging.ERROR)

# Load the saved model and scaler
model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# =========================================
# 🌟 App Title and Description
# =========================================
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")

st.title("🩸 Diabetes Prediction System")
st.markdown("""
This AI-powered web app predicts the likelihood of diabetes based on patient health details.  
It uses an **XGBoost model** trained with gender-specific adjustments (males have no pregnancies).
""")

st.divider()

# =========================================
# 🧍 User Input Form
# =========================================
st.header("Enter Patient Details")

# Gender Input
gender_input = st.radio("Gender:", ["Male", "Female"])

if gender_input == "Male":
    gender = 1
    pregnancies = 0
else:
    gender = 0
    pregnancies = st.number_input("Number of Pregnancies:", min_value=0, max_value=20, step=1)

# Other Inputs
glucose = st.number_input("Glucose Level (mg/dL):", min_value=0.0, step=1.0)
blood_pressure = st.number_input("Blood Pressure (mm Hg):", min_value=0.0, step=1.0)
skin_thickness = st.number_input("Skin Thickness (mm):", min_value=0.0, step=1.0)
bmi = st.number_input("Body Mass Index (BMI):", min_value=0.0, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function:", min_value=0.0, step=0.01)
age = st.number_input("Age:", min_value=1, max_value=120, step=1)

# =========================================
# 🧮 Prediction
# =========================================
if st.button("🔍 Predict Diabetes"):
    # Create DataFrame
    user_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, bmi,
                               dpf, age, gender]],
                             columns=['Pregnancies', 'Glucose', 'BloodPressure',
                                      'SkinThickness', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Gender'])

    # Scale input
    user_data_scaled = scaler.transform(user_data)

    # Predict
    prediction = model.predict(user_data_scaled)[0]
    probability = model.predict_proba(user_data_scaled)[0][1]

    # =========================================
    # 🩺 Display Result
    # =========================================
    st.divider()
    st.subheader("Prediction Result:")

    if prediction == 1:
        st.error(f"🩸 The person is **LIKELY to have Diabetes** (Confidence: {probability*100:.2f}%)")
    else:
        st.success(f"✅ The person is **NOT likely to have Diabetes** (Confidence: {(1-probability)*100:.2f}%)")

    st.progress(float(probability))

st.divider()
st.caption("⚙️ Model: XGBoost | Data Source: Pima Indians Diabetes Dataset | Developer: Podmaraj Boruah")
