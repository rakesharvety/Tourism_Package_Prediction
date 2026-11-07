"""Streamlit app: predict whether a customer will buy the Wellness Tourism Package.

Loads the model that the GitHub Actions pipeline trained and committed next to
this file, collects customer details into a DataFrame, and shows the prediction.
Run on Streamlit Community Cloud with main file path tourism_project/deployment/app.py
"""
import os

import joblib
import pandas as pd
import streamlit as st

# set_page_config must be the first Streamlit command in the script
st.set_page_config(page_title="Wellness Tourism Package Prediction", page_icon="🧳")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("Wellness Tourism Package Prediction")
st.write(
    "Enter a customer's details to estimate how likely they are to purchase the "
    "Wellness Tourism Package. Use the result to prioritise whom to contact."
)

# --- Customer details --------------------------------------------------------
st.subheader("Customer details")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 100, 35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox(
        "Marital status", ["Married", "Single", "Divorced", "Unmarried"]
    )
    occupation = st.selectbox(
        "Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"]
    )
    designation = st.selectbox(
        "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    monthly_income = st.number_input("Monthly income", 1000.0, 200000.0, 23000.0, 500.0)
    city_tier = st.selectbox("City tier", [1, 2, 3])
with col2:
    passport = st.selectbox("Holds a valid passport?", ["No", "Yes"])
    own_car = st.selectbox("Owns a car?", ["No", "Yes"])
    num_persons = st.number_input("Number of people visiting", 1, 10, 3)
    num_children = st.number_input("Children below age 5 visiting", 0, 5, 0)
    num_trips = st.number_input("Average trips per year", 0.0, 30.0, 3.0, 1.0)
    preferred_star = st.selectbox("Preferred property star rating", [3.0, 4.0, 5.0])

# --- Interaction details -----------------------------------------------------
st.subheader("Sales interaction")
col3, col4 = st.columns(2)
with col3:
    type_of_contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
    product_pitched = st.selectbox(
        "Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
    )
    duration_of_pitch = st.number_input("Duration of pitch (minutes)", 1.0, 150.0, 15.0, 1.0)
with col4:
    num_followups = st.number_input("Number of follow-ups", 0.0, 10.0, 3.0, 1.0)
    pitch_score = st.slider("Pitch satisfaction score", 1, 5, 3)

# Inputs saved into a DataFrame with the same column names used in training
input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_persons,
    "NumberOfFollowups": num_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": pitch_score,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

with st.expander("Input data sent to the model"):
    st.dataframe(input_df)

if st.button("Predict"):
    probability = float(model.predict_proba(input_df)[0, 1])
    prediction = int(model.predict(input_df)[0])
    st.subheader("Prediction result")
    if prediction == 1:
        st.success(f"Likely to purchase the Wellness Tourism Package ({probability:.1%} probability).")
    else:
        st.warning(f"Unlikely to purchase the Wellness Tourism Package ({probability:.1%} probability).")
