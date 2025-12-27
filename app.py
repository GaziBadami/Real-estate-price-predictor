import pickle
import streamlit as st
import numpy as np
import pandas as pd

# Load model
model = pickle.load(open("house_prices_model.pickle", "rb"))

# Load correct column names
model_columns = pickle.load(open("model_columns.pkl", "rb"))

st.title("🏠 Real Estate Price Prediction App")

location = st.selectbox("Location", options=[c for c in model_columns if c not in ["Carpet Area", "Bathroom", "Bhk"]])
carpet_area = st.number_input("Carpet Area", min_value=300, max_value=5000)
bathroom = st.number_input("Bathroom", min_value=1, max_value=10)
bhk = st.number_input("BHK", min_value=1, max_value=10)

if st.button("Predict Price"):
    input_df = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)

    input_df["Carpet Area"] = carpet_area
    input_df["Bathroom"] = bathroom
    input_df["Bhk"] = bhk

    if location in input_df.columns:
        input_df[location] = 1

    price = model.predict(input_df)[0]
    price_in_rupees = round(price * 100000, 2)

    st.success(f"💰 Estimated Price: ₹ {price_in_rupees:,.2f}")
