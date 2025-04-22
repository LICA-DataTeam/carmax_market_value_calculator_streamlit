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

import streamlit as st # type: ignore
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
col1, col2, col3, col4 = st.columns(4)

for brand in brands:
    models = df_model[df_model["BRANDS"] == brand]["MODEL"].values
    for model in models:
        calculator1 = mvc.MarketValueCalculator(df_bq, loss_method = "MAE", verbose=True, truncate_upper=1.0, truncate_lower=0.0)
        calculator1.car_brand = brand
        calculator1.car_model = model
        calculator1.car_year = 2025
        calculator1.car_variant_price = None
        calculator1.initialize_optimization()
        # response1 = calculator1.calculate_market_value(brand, model, 2025, car_variant_price=None)
        calculator2 = mvc.MarketValueCalculator(df_bq, loss_method = "MAE", verbose=True, truncate_upper=0.98, truncate_lower=0.1)
        calculator2.car_brand = brand
        calculator2.car_model = model
        calculator2.car_year = 2025
        calculator2.car_variant_price = None
        calculator2.initialize_optimization()
        # response2 = calculator2.calculate_market_value(brand, model, 2025, car_variant_price=None)

        with col1:
            fig1, ax1 = plt.subplots()
            x1 = calculator1.df_train["car_years_old"]
            y1 = calculator1.df_train["price"]
            ax1.scatter(x1, y1, alpha=0.1)
            ax1.set_xlim(-0.5, 20.5)
            ax1.set_title(f"{brand} {model}")
            ax1.set_xlabel("Year")
            ax1.set_ylabel("Price")
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots()
            x2 = calculator1.df_train["car_years_old"]
            y2 = calculator1.df_train["price_filter"]
            ax2.scatter(x2, y2, alpha=0.1)
            ax2.set_xlim(ax1.get_xlim())
            ax2.set_ylim(ax1.get_ylim())
            ax2.set_title(f"{brand} {model}")
            ax2.set_xlabel("Year")
            ax2.set_ylabel("Price")
            st.pyplot(fig2)

        with col3:
            fig3, ax3 = plt.subplots()
            x3 = calculator2.df_train["car_years_old"]
            y3 = calculator2.df_train["price_filter"]
            ax3.scatter(x3, y3, alpha=0.1)
            ax3.set_xlim(ax1.get_xlim())
            ax3.set_ylim(ax1.get_ylim())
            ax3.set_title(f"{brand} {model}")
            ax3.set_xlabel("Year")
            ax3.set_ylabel("Price")
            st.pyplot(fig3)

        with col4:
            fig4, ax4 = plt.subplots()
            x4 = calculator2.df_train["car_years_old"]
            y4 = calculator2.df_train["price"]
            ax4.scatter(x4, y4, alpha=0.1)
            ax4.set_xlim(ax1.get_xlim())
            ax4.set_ylim(ax1.get_ylim())
            ax4.set_title(f"{brand} {model}")
            ax4.set_xlabel("Year")
            ax4.set_ylabel("Price")
            st.pyplot(fig4)