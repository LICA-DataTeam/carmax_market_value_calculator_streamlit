"""
Carmax Market Value Calculator
==============================

written from March 20, 2025 to april 21, 2025
created by JP

Overview:
This script provides a Streamlit-based web application for calculating the market value of cars.
Users can select car details such as year, brand, model, and variant price to estimate the car's market value.

Features:
- Interactive UI for selecting car details.
- Integration with BigQuery and Google Sheets for data retrieval.
- Dynamic market value calculation based on user inputs.
- Visualization of market value trends using Streamlit.

Modules:
- Streamlit: Provides the web application interface.
- NumPy: Handles numerical operations.
- market_value_app_utils: Contains utility functions for data initialization.
- market_value_app_calculator: Implements the market value calculation logic.

Functions:
- initialize_data_cache(): Caches data from BigQuery and Google Sheets to avoid multiple request of the same data.
- Main Streamlit UI: Allows users to input car details and displays the calculated market value.
"""

import streamlit as st
import app_utils as app_utils
import app_calculator as mvc
import numpy as np
import matplotlib.pyplot as plt

@st.cache_resource
def initialize_data_cache(bq_acct):
    """
    Cache data from BigQuery and Google Sheets for faster access.

    Returns:
    - brands (list): List of car brands.
    - df_model (pd.DataFrame): DataFrame containing car models and their brands.
    - years (list): List of car manufacturing years.
    - df_bq (pd.DataFrame): DataFrame containing car listings data from BigQuery.
    """
    return app_utils.initialize_data(bq_acct)

st.title("Carmax Market Value Calculator")
brands, df_model, years, df_bq = initialize_data_cache(st.secrets["bq_acct"].to_dict())

# User input for car details
year = st.selectbox("Car year", np.append(["Select car year"], years))
brand = st.selectbox("Car brand", np.append(["Select car brand"], brands))

if brand != "Select car brand":
    df_brand_model = df_model[df_model["BRANDS"] == brand]["MODEL"]
    model = st.selectbox("Car model", np.append(["Select car model"], df_brand_model.values))
else:
    model = st.selectbox("Car model", [])

# Placeholder for car variant
variant_price = st.selectbox("Car variant", [])
st.warning("Car variant is currently under development.")
variant_price = st.number_input("Car variant price", value=None, step=1)
st.info("For the meantime, search the brand new price of the variant and put it above.")

# Calculate market value if all inputs are provided
if year != "Select car year" and brand != "Select car brand" and model != "Select car model":
    calculator = mvc.MarketValueCalculator(df_bq, loss_method = "MAE", verbose=True, truncate_upper=0.98, truncate_lower=0.1)
    response = calculator.calculate_market_value(brand, model, year, car_variant_price=variant_price)
    st.header(f"{brand} {model} {year} market value: {response['market_value']}")
    if calculator.verbose:
        st.pyplot(calculator.fig)