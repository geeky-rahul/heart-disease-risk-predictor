import streamlit as st
import pandas as pd
import joblib

# Load model and tools
model = joblib.load('heart_disease_logistic_model.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

# Page setup
st.set_page_config(page_title="Heart Disease Risk Predictor", layout="centered", initial_sidebar_state="collapsed")

# Title and intro
st.markdown("""
# Heart Disease Risk Predictor

This application uses medical data and machine learning to estimate your heart disease risk.

Please fill out the form below with accurate information. 
""")

st.markdown("---")

# Form layout
with st.form("heart_check_form"):
    st.subheader("Basic Information")
    age = st.slider("Your Age", 18, 100, 45)
    sex = st.radio("Gender", ["Male", "Female"], horizontal=True)

    st.subheader("Health Measurements")
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol Level (mg/dL)", 100, 600, 200)
    fasting_bs = st.radio("Is your fasting blood sugar above 120?", [1, 0],
                          format_func=lambda x: "Yes" if x == 1 else "No")

    max_hr = st.slider("Your Maximum Heart Rate", 60, 220, 150)
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)

    st.subheader("Other Details")
    chest_pain = st.selectbox("Type of Chest Pain", ["ATA", "NAP", "TA", "ASY"])
    resting_ecg = st.selectbox("ECG Result", ["Normal", "ST", "LVH"])
    
    # ✅ Keep Yes/No for user, map internally for model
    exercise_angina = st.radio("Do you experience chest pain during physical activity (Exercise Angina)?", ["Yes", "No"], horizontal=True)
    exercise_angina = "Y" if exercise_angina == "Yes" else "N"

    st_slope = st.selectbox("Slope of ST Segment", ["Up", "Flat", "Down"])

    submitted = st.form_submit_button("Predict My Risk")

# On submit
if submitted:
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    # st.markdown("---")
    st.subheader("Your Result")

    if prediction == 1:
        st.error("⚠️ You may be at **higher risk** of heart-related issues.\n\nPlease consider talking to a doctor for a full health checkup.")
    else:
        st.success("✅ You appear to be at **low risk** of heart problems.\n\nKeep maintaining a healthy lifestyle!")

    st.caption("Note: This tool gives an estimate only. For accurate diagnosis, please consult a medical professional.")

