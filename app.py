
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Tourism Package Predictor", page_icon="", layout="centered")

@st.cache_resource
def load_model():
return joblib.load("models/best_model.pkl")

model = load_model()

st.title("Wellness Tourism Package Predictor")
st.write("Predict whether a customer is likely to purchase the tourism package.")

age = st.number_input("Age", min_value=18, max_value=100, value=35)
typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input("Duration of Pitch", min_value=0.0, value=15.0)
occupation = st.selectbox("Occupation", ["Free Lancer", "Large Business", "Salaried", "Small Business"])
gender = st.selectbox("Gender", ["Female", "Male"])
number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, value=2)
number_of_followups = st.number_input("Number of Follow-ups", min_value=0, value=3)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"])
preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Divorced", "Married", "Single", "Unmarried"])
number_of_trips = st.number_input("Number of Trips per Year", min_value=0, value=2)
passport = st.selectbox("Passport Available", [0, 1])
own_car = st.selectbox("Own Car", [0, 1])
number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, value=0)
designation = st.selectbox("Designation", ["AVP", "Executive", "Manager", "Senior Manager", "VP"])
monthly_income = st.number_input("Monthly Income", min_value=0.0, value=30000.0)
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)

if st.button("Predict Purchase"):
input_df = pd.DataFrame([{
"Age": age,
"TypeofContact": typeof_contact,
"CityTier": city_tier,
"DurationOfPitch": duration_of_pitch,
"Occupation": occupation,
"Gender": gender,
"NumberOfPersonVisiting": number_of_person_visiting,
"NumberOfFollowups": number_of_followups,
"ProductPitched": product_pitched,
"PreferredPropertyStar": preferred_property_star,
"MaritalStatus": marital_status,
"NumberOfTrips": number_of_trips,
"Passport": passport,
"PitchSatisfactionScore": pitch_satisfaction_score,
"OwnCar": own_car,
"NumberOfChildrenVisiting": number_of_children_visiting,
"Designation": designation,
"MonthlyIncome": monthly_income
}])

probability = model.predict_proba(input_df)[0, 1]
prediction = model.predict(input_df)[0]

st.subheader("Prediction Result")
st.write(f"Purchase probability: {probability:.2%}")

if prediction == 1:
st.success("Likely to purchase the Wellness Tourism Package.")
else:
st.warning("Less likely to purchase the Wellness Tourism Package.")
