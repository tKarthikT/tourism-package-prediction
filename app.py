import streamlit as st
import joblib
import pandas as pd

st.title("Tourism Package Prediction")

@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")

model = load_model()

st.write("Enter customer details to predict tourism package purchase:")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
monthly_income = st.number_input("Monthly Income", min_value=0, value=20000)
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, value=10)
number_of_trips = st.number_input("Number of Trips per Year", min_value=0, value=2)
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
number_of_followups = st.number_input("Number of Followups", min_value=0, value=2)

if st.button("Predict"):
    try:
        input_df = pd.DataFrame({
            "Age": [age],
            "MonthlyIncome": [monthly_income],
            "DurationOfPitch": [duration_of_pitch],
            "NumberOfTrips": [number_of_trips],
            "PitchSatisfactionScore": [pitch_satisfaction_score],
            "NumberOfFollowups": [number_of_followups]
        })
        prediction = model.predict(input_df)
        st.success(f"Prediction: {'Will Purchase' if prediction[0] == 1 else 'Will Not Purchase'}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Note: Model may expect a different feature set. Please check the training pipeline's feature columns.")
