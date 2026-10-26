
import streamlit as st
import requests

API_URL = "https://deputize-plated-entangled.ngrok-free.dev"

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Risk Predictor")
st.markdown("Enter patient details to predict diabetes risk.")
st.markdown("---")

# Input fields
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies", min_value=0, max_value=17, value=1
    )

    glucose = st.slider(
        "Glucose", 0.0, 200.0, 120.0, 1.0
    )

    blood_pressure = st.slider(
        "Blood Pressure", 0.0, 130.0, 70.0, 1.0
    )

    skin_thickness = st.slider(
        "Skin Thickness", 0.0, 100.0, 20.0, 1.0
    )

with col2:
    insulin = st.slider(
        "Insulin", 0.0, 850.0, 79.0, 1.0
    )

    bmi = st.slider(
        "BMI", 0.0, 70.0, 25.0, 0.1
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=2.5,
        value=0.5,
        step=0.01
    )

    age = st.slider(
        "Age", 18, 90, 30
    )

st.markdown("---")

# Predict button
if st.button("Predict Diabetes Risk", use_container_width=True):

    payload = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }

    try:
        res = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=15
        )

        if res.status_code == 200:
            result = res.json()

            if result["result_label"] == "Diabetic":
                st.error("Result: Diabetic")
            else:
                st.success("Result: Not Diabetic")

            st.metric(
                "Confidence",
                f"{result['confidence'] * 100:.1f}%"
            )

            with st.expander("Raw API Response"):
                st.json(result)

        else:
            st.error(
                "Prediction failed. Please check that "
                "FastAPI is running correctly."
            )

    except requests.exceptions.RequestException:
        st.error(
            "Cannot connect to FastAPI. "
            "Please start the FastAPI server first."
        )

st.markdown("---")
st.caption(
    "Educational prediction only. "
    "This model is not a medical diagnosis."
)
