import pandas as pd
import numpy as np
import psycopg2
from psycopg2 import Error
import os
from sqlalchemy import create_engine, text
from sqlalchemy.types import String, Integer, Date, Numeric

# --- Centralized Configuration ---
# CSV file path (Option 1: relative to script)
CSV_FILE_PATH_OPTION1 = os.path.join(os.path.dirname(__file__), '..', 'sales_data.csv')
# CSV file path (Option 2: absolute path - USE IF OPTION 1 FAILS)
CSV_FILE_PATH_OPTION2 = r"C:\Users\Vishnu B M\Documents\Power BI Projects\Sales Forecast and Demand Planning\sales_data.csv"

# Auto-detect CSV_FILE_PATH
CSV_FILE_PATH = None
if os.path.exists(CSV_FILE_PATH_OPTION1):
    CSV_FILE_PATH = CSV_FILE_PATH_OPTION1
    print(f"Using CSV path (Option 1): {CSV_FILE_PATH}")
elif os.path.exists(CSV_FILE_PATH_OPTION2):
    CSV_FILE_PATH = CSV_FILE_PATH_OPTION2
    print(f"Using CSV path (Option 2): {CSV_FILE_PATH}")
else:
    print(f"Error: sales_data.csv not found at {CSV_FILE_PATH_OPTION1} or {CSV_FILE_PATH_OPTION2}.")
    print("Please ensure the CSV file is in the correct directory.")
    print("If your project structure is different, you might need to adjust CSV_FILE_PATH.")
    exit(1) # Exit the script if CSV is not found

# Database connection details
DB_HOST = "localhost"
DB_NAME = "retail_sales_db"
DB_USER = "retail_user" # <--- REPLACE WITH YOUR ACTUAL USERNAME
DB_PASSWORD = "salesforecast" # <--- REPLACE WITH YOUR ACTUAL PASSWORD
DB_PORT = "5432"

# Table names
RAW_TABLE_NAME = "raw_sales"
CLEANED_TABLE_NAME = "cleaned_sales"
# --- End Centralized Configuration ---


def get_db_engine():
    """Establishes and returns a SQLAlchemy PostgreSQL database engine."""
    db_connection_str = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    try:
        engine = create_engine(db_connection_str)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        print("Successfully connected to PostgreSQL via SQLAlchemy engine.")
        return engine
    except Exception as e:
        print(f"Error connecting to PostgreSQL via SQLAlchemy: {e}")
        print("Please ensure PostgreSQL server is running and connection details are correct (host, port, user, password, database name).")
        return None

def ingest_raw_csv_to_db(engine):
    """
    Ingests raw sales data from CSV into a PostgreSQL table.
    """
    print(f"\n--- Starting Raw Data Ingestion from {CSV_FILE_PATH} to '{RAW_TABLE_NAME}' ---")
    try:
        df_raw = pd.read_csv(CSV_FILE_PATH)
        print(f"Successfully loaded {len(df_raw)} rows from CSV.")
        print(f"Original CSV columns: {df_raw.columns.tolist()}")

        # Convert 'Date' column to datetime objects
        df_raw['Date'] = pd.to_datetime(df_raw['Date']).dt.date # Store as date only for DB

        # Define column types for SQL to ensure proper schema
        raw_column_types = {
            'Date': Date,
            'Store ID': String(50),
            'Product ID': String(50),
            'Category': String(100),
            'Region': String(100),
            'Inventory Level': Integer,
            'Units Sold': Integer,
            'Units Ordered': Integer,
            'Price': Numeric(10, 2),
            'Discount': Numeric(5, 2),
            'Weather Condition': String(100),
            'Promotion': String(100),
            'Competitor Pricing': Numeric(10, 2),
            'Seasonality': String(100),
            'Epidemic': String(100),
            'Demand': String(100)
        }

        with engine.connect() as connection:
            connection.execute(text(f"DROP TABLE IF EXISTS {RAW_TABLE_NAME} CASCADE;"))
            connection.commit()
            print(f"Cleared existing data in '{RAW_TABLE_NAME}' (if it existed).")

            print(f"Attempting to insert {len(df_raw)} rows into '{RAW_TABLE_NAME}'...")
            df_raw.to_sql(RAW_TABLE_NAME, engine, if_exists='append', index=False, dtype=raw_column_types)
            print(f"Raw data successfully ingested into '{RAW_TABLE_NAME}' table in PostgreSQL.")
        return True
    except Exception as e:
        print(f"An error occurred during raw data ingestion: {e}")
        print("Raw data ingestion failed. Skipping basic data cleaning.")
        return False

def clean_and_save_data(engine):
    """
    Loads raw data, performs basic cleaning, and saves to a new PostgreSQL table.
    """
    print(f"\n--- Starting Basic Data Cleaning from '{RAW_TABLE_NAME}' to '{CLEANED_TABLE_NAME}' ---")
    try:
        # Load raw data from DB
        df_cleaned = pd.read_sql_table(RAW_TABLE_NAME, engine)
        print(f"Loaded {len(df_cleaned)} rows from '{RAW_TABLE_NAME}' for cleaning.")
        print(f"Columns immediately after loading from raw_sales: {df_cleaned.columns.tolist()}")

        # Convert column names to lowercase and replace spaces with underscores
        # Example: 'Store ID' -> 'store_id'
        df_cleaned.columns = df_cleaned.columns.str.replace(' ', '_').str.lower()
        print(f"Columns after standardizing (spaces to underscores, then lowercasing): {df_cleaned.columns.tolist()}")

        # Basic cleaning: Handle potential missing values or inconsistencies
        for col in ['units_sold', 'units_ordered', 'price', 'discount', 'competitor_pricing', 'inventory_level']:
            if col in df_cleaned.columns:
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').fillna(0)
        for col in ['category', 'region', 'weather_condition', 'promotion', 'seasonality', 'epidemic', 'demand']:
             if col in df_cleaned.columns:
                 df_cleaned[col] = df_cleaned[col].astype(str).fillna('Unknown')

        # Define column types for SQL to ensure proper schema for cleaned_sales
        # These must match the standardized column names
        cleaned_column_types = {
            'date': Date,
            'store_id': String(50),
            'product_id': String(50),
            'category': String(100),
            'region': String(100),
            'inventory_level': Integer,
            'units_sold': Integer,
            'units_ordered': Integer,
            'price': Numeric(10, 2),
            'discount': Numeric(5, 2),
            'weather_condition': String(100),
            'promotion': String(100),
            'competitor_pricing': Numeric(10, 2),
            'seasonality': String(100),
            'epidemic': String(100),
            'demand': String(100)
        }


        with engine.connect() as connection:
            connection.execute(text(f"DROP TABLE IF EXISTS {CLEANED_TABLE_NAME} CASCADE;"))
            connection.commit()
            print(f"Existing table '{CLEANED_TABLE_NAME}' dropped (if it existed).")

            df_cleaned.to_sql(CLEANED_TABLE_NAME, engine, if_exists='append', index=False, dtype=cleaned_column_types, method='multi')
            print(f"Cleaned data successfully saved to '{CLEANED_TABLE_NAME}' table in PostgreSQL.")
        return True
    except Exception as e:
        print(f"An error occurred during data cleaning or saving: {e}")
        print("Basic data cleaning failed. Check logs for details.")
        return False

if __name__ == "__main__":
    print("--- Running 01_ingest_and_clean_data.py - Version 2025-07-30_D ---") # NEW VERSION IDENTIFIER
    engine = None
    try:
        engine = get_db_engine()
        if engine is None:
            print("Could not establish database engine connection. Exiting pipeline.")
        else:
            if ingest_raw_csv_to_db(engine):
                clean_and_save_data(engine)
    finally:
        print("\nPostgreSQL connection closed.")