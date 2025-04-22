"""
Market Value App Utilities
==========================

written from March 20, 2025 to april 21, 2025
created by JP

Overview:
This script contains utility functions for the Carmax Market Value Calculator app.
It handles data retrieval from BigQuery and Google Sheets and prepares the data for use in the application.

Features:
- Retrieve car listings data from BigQuery.
- Retrieve car brand and model data from Google Sheets.
- Initialize and preprocess data for the application.

Modules:
- Pandas: Handles data manipulation and processing.
- bq_utils: Manages data retrieval from BigQuery.

Functions:
- load_data_from_bq(): Retrieves car listings data from BigQuery.
- load_data_from_sheet(): Retrieves car brand and model data from Google Sheets.
- initialize_data(): Combines data from BigQuery and Google Sheets and prepares it for use.
"""

import pandas as pd
import bq_utils

def load_data_from_bq(bq_acct):
    """
    Retrieve car listings data from BigQuery.

    Returns:
    - df_bq (pd.DataFrame): DataFrame containing car listings data.
    """
    print("Loading data from BigQuery...")
    sql_query = """
        SELECT
        product_title AS car_description,
        car_year,
        km_driven AS distance_driven,
        price,
        FROM `absolute-gantry-363408.carmax_webscrape.all_cars_ph`

        UNION ALL

        SELECT
        car_model AS car_description,
        car_year,
        distance_km AS distance_driven,
        IF(price < 3000, 1000*price, price) AS price,
        FROM `absolute-gantry-363408.carmax_webscrape.fb_marketplace`
        WHERE 100000 < price AND price < 3000000
        """
    df_bq = bq_utils.sql_query_bq(sql_query, bq_acct)
    return df_bq

def load_data_from_sheet():
    """
    Retrieve car brand and model data from Google Sheets.

    Returns:
    - df_sheet (pd.DataFrame): DataFrame containing car brand and model data.
    """
    print("Loading data from Google Sheets...")
    sheet_name = "car_models"
    sheet_id = "1LSsePyDiEpaBgnUoa2XWCJ-R8m3-bLSxNq0plImrtEs"
    sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    df_sheet = pd.read_csv(sheet_url)
    return df_sheet

def initialize_data(bq_acct):
    """
    Initialize and preprocess data for the application.

    Returns:
    - brand (list): List of car brands.
    - df_model (pd.DataFrame): DataFrame containing car models and their brands.
    - year (list): List of car manufacturing years.
    - df_bq (pd.DataFrame): DataFrame containing car listings data.
    """
    df_sheet = load_data_from_sheet()
    df_bq = load_data_from_bq(bq_acct)

    brand = df_sheet["BRANDS"].unique()
    df_model = df_sheet[["BRANDS", "MODEL"]]
    year = [2025 - i for i in range(26)]
    return brand, df_model, year, df_bq
