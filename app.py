# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------
# Load trained model & encoders
# -----------------------------
with open('model.pkl', 'rb') as f:
    rf = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# -----------------------------
# Feature configuration
# -----------------------------
feature_cols = [
    'Gr Liv Area', 'Overall Qual', 'Overall Cond', 'Year Built', 'Year Remod/Add',
    '1st Flr SF', '2nd Flr SF', 'Full Bath', 'Half Bath', 'Bedroom AbvGr',
    'Kitchen AbvGr', 'TotRms AbvGrd', 'Fireplaces', 'Pool Area', 'Wood Deck SF',
    'Open Porch SF', 'Enclosed Porch', '3Ssn Porch', 'Screen Porch', 'Lot Area',
    'MS Zoning', 'Neighborhood', 'House Style', 'Bldg Type', 'Exterior 1st', 'Exterior 2nd'
]

# Friendly names & tooltips
friendly_names = {
    'Gr Liv Area': 'Above Ground Living Area (sq ft)',
    'Overall Qual': 'Overall Material & Finish Quality',
    'Overall Cond': 'Overall Condition Rating',
    'Year Built': 'Year Built',
    'Year Remod/Add': 'Year of Remodel/Additions',
    '1st Flr SF': '1st Floor Area (sq ft)',
    '2nd Flr SF': '2nd Floor Area (sq ft)',
    'Full Bath': 'Number of Full Bathrooms',
    'Half Bath': 'Number of Half Bathrooms',
    'Bedroom AbvGr': 'Bedrooms Above Ground',
    'Kitchen AbvGr': 'Kitchens Above Ground',
    'TotRms AbvGrd': 'Total Rooms Above Ground',
    'Fireplaces': 'Number of Fireplaces',
    'Pool Area': 'Pool Area (sq ft)',
    'Wood Deck SF': 'Wood Deck Area (sq ft)',
    'Open Porch SF': 'Open Porch Area (sq ft)',
    'Enclosed Porch': 'Enclosed Porch Area (sq ft)',
    '3Ssn Porch': 'Three Season Porch Area (sq ft)',
    'Screen Porch': 'Screened Porch Area (sq ft)',
    'Lot Area': 'Lot Area (sq ft)',
    'MS Zoning': 'Zoning Classification',
    'Neighborhood': 'Neighborhood',
    'House Style': 'House Style',
    'Bldg Type': 'Building Type',
    'Exterior 1st': 'Exterior Covering 1',
    'Exterior 2nd': 'Exterior Covering 2'
}

# -----------------------------
# Streamlit app
# -----------------------------
st.title("🏠 Ames Housing Price Predictor")
st.write("Fill in the details of the house below to predict its sale price.")

user_input = {}

# -----------------------------
# Numeric features
# -----------------------------
numeric_cols = [col for col in feature_cols if col not in label_encoders]
for col in numeric_cols:
    user_input[col] = st.number_input(
        label=friendly_names[col],
        value=0.0,
        step=1.0,
        help=friendly_names[col]
    )

# -----------------------------
# Categorical features
# -----------------------------
categorical_cols = [col for col in feature_cols if col in label_encoders]
for col in categorical_cols:
    user_input[col] = st.selectbox(
        label=friendly_names[col],
        options=list(label_encoders[col].classes_),
        index=0,
        help=friendly_names[col]
    )

# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict Sale Price"):
    # Encode categorical features
    for col in categorical_cols:
        user_input[col] = label_encoders[col].transform([user_input[col]])[0]

    # Convert to DataFrame
    input_df = pd.DataFrame([user_input], columns=feature_cols)

    # Make prediction
    prediction = rf.predict(input_df)[0]

    st.success(f"💰 Predicted Sale Price: ${prediction:,.2f}")
