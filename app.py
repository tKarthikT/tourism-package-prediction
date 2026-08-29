import streamlit as st
import joblib
import pandas as pd

st.title("Tourism Package Prediction")

@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")

model = load_model()

st.write("Enter customer details to predict tourism package purchase:")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, value=10)
    number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, value=2)
    number_of_followups = st.number_input("Number of Followups", min_value=0, value=2)
    preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
    number_of_trips = st.number_input("Number of Trips per Year", min_value=0, value=2)
    passport = st.selectbox("Has Passport", [0, 1])
    monthly_income = st.number_input("Monthly Income", min_value=0, value=20000)

with col2:
    type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
    occupation = st.selectbox("Occupation", ["Free Lancer", "Large Business", "Salaried", "Small Business"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"])
    marital_status = st.selectbox("Marital Status", ["Divorced", "Married", "Single", "Unmarried"])
    designation = st.selectbox("Designation", ["AVP", "Executive", "Manager", "Senior Manager", "VP"])
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    own_car = st.selectbox("Owns Car", [0, 1])
    number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, value=0)

if st.button("Predict"):
    try:
        input_df = pd.DataFrame({
            "Age": [age],
            "TypeofContact": [type_of_contact],
            "CityTier": [city_tier],
            "DurationOfPitch": [duration_of_pitch],
            "Occupation": [occupation],
            "Gender": [gender],
            "NumberOfPersonVisiting": [number_of_person_visiting],
            "NumberOfFollowups": [number_of_followups],
            "ProductPitched": [product_pitched],
            "PreferredPropertyStar": [preferred_property_star],
            "MaritalStatus": [marital_status],
            "NumberOfTrips": [number_of_trips],
            "Passport": [passport],
            "PitchSatisfactionScore": [pitch_satisfaction_score],
            "OwnCar": [own_car],
            "NumberOfChildrenVisiting": [number_of_children_visiting],
            "Designation": [designation],
            "MonthlyIncome": [monthly_income]
        })
        prediction = model.predict(input_df)
        result = "Will Purchase" if prediction[0] == 1 else "Will Not Purchase"
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
