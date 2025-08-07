import pandas as pd
import numpy as np
import psycopg2
from psycopg2 import Error
import os
from sqlalchemy import create_engine, text
from sqlalchemy.types import String, Integer, Date, Numeric

# --- Centralized Configuration ---
# Database connection details
DB_HOST = "localhost"
DB_NAME = "retail_sales_db"
DB_USER = "retail_user" # <--- REPLACE WITH YOUR ACTUAL USERNAME
DB_PASSWORD = "salesforecast" # <--- REPLACE WITH YOUR ACTUAL PASSWORD
DB_PORT = "5432"

# Table names
CLEANED_TABLE_NAME = "cleaned_sales"
PROCESSED_TABLE_NAME = "processed_sales"
# --- End Centralized Configuration ---


def get_db_engine():
    """Establishes and returns a SQLAlchemy PostgreSQL database engine."""
    db_connection_str = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    try:
        engine = create_engine(db_connection_str)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        print("Successfully connected to PostgreSQL via SQLAlchemy engine for feature engineering.")
        return engine
    except Exception as e:
        print(f"Error connecting to PostgreSQL via SQLAlchemy: {e}")
        print("Please ensure PostgreSQL server is running and connection details are correct (host, port, user, password, database name).")
        return None

def load_cleaned_data(engine):
    """
    Loads cleaned sales data from PostgreSQL.
    """
    print(f"\n--- Loading cleaned data from '{CLEANED_TABLE_NAME}' ---")
    try:
        query = f"SELECT * FROM {CLEANED_TABLE_NAME} ORDER BY date;"
        df = pd.read_sql_query(text(query), engine, parse_dates=['date'])
        # Ensure all column names are lowercase right after loading
        original_columns = df.columns.tolist()
        df.columns = df.columns.str.lower()
        print(f"Loaded {len(df)} rows from '{CLEANED_TABLE_NAME}'.")
        print(f"Original columns from DB: {original_columns}")
        print(f"Columns after lowercasing: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error loading cleaned data: {e}")
        return None

def engineer_features(df):
    """
    Engineers new features from the cleaned sales data.
    """
    print("\n--- Starting Feature Engineering ---")
    if df.empty:
        print("No data to engineer features from. Exiting feature engineering.")
        return pd.DataFrame()

    # Debugging print to show columns at the start of engineer_features
    print(f"Columns at entry of engineer_features: {df.columns.tolist()}")

    df['date'] = pd.to_datetime(df['date'])

    # --- EXPLICIT AND IMMEDIATE CHECK FOR store_id AND product_id ---
    # If these are missing, we cannot proceed with core features.
    required_ids = ['store_id', 'product_id']
    for col in required_ids:
        if col not in df.columns:
            print(f"FATAL ERROR: Required identifier column '{col}' not found in DataFrame.")
            print(f"Current DataFrame columns: {df.columns.tolist()}")
            return pd.DataFrame() # Stop early if core IDs are missing

    # Create store_product_id immediately and verify its creation
    try:
        df['store_product_id'] = df['store_id'] + '_' + df['product_id']
        print("Successfully created 'store_product_id'.")
        if 'store_product_id' not in df.columns:
            print("WARNING: 'store_product_id' was supposedly created but not found in columns!")
            return pd.DataFrame() # Something went wrong if it's not there
    except Exception as e:
        print(f"ERROR: Failed to create 'store_product_id': {e}")
        return pd.DataFrame()

    # Sort data immediately after creating the compound ID
    df = df.sort_values(by=['store_product_id', 'date'])
    print("Data sorted by 'store_product_id' and 'date'.")

    # Time-based features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
    print("Successfully created time-based features.")
    print(f"Columns after time-based features: {df.columns.tolist()}") # Debugging print


    # Units Sold Lag Features (only if units_sold is present)
    if 'units_sold' in df.columns:
        df['units_sold_lag_1'] = df.groupby('store_product_id')['units_sold'].shift(1)
        df['units_sold_lag_7'] = df.groupby('store_product_id')['units_sold'].shift(7)
        print("Successfully created 'units_sold' lag features.")
    else:
        print("Warning: 'units_sold' column not found, skipping lag features.")

    # Units Sold Rolling Mean Features (only if units_sold is present)
    if 'units_sold' in df.columns:
        df['units_sold_rolling_mean_7'] = df.groupby('store_product_id')['units_sold'].transform(lambda x: x.rolling(window=7, min_periods=1).mean().shift(1))
        df['units_sold_rolling_mean_30'] = df.groupby('store_product_id')['units_sold'].transform(lambda x: x.rolling(window=30, min_periods=1).mean().shift(1))
        print("Successfully created 'units_sold' rolling mean features.")
    else:
        print("Warning: 'units_sold' column not found, skipping rolling mean features.")

    # Fill NaN values created by shift/rolling operations
    for col in ['units_sold_lag_1', 'units_sold_lag_7', 'units_sold_rolling_mean_7', 'units_sold_rolling_mean_30']:
        if col in df.columns: # Only fill if the column was actually created
            df[col] = df[col].fillna(0)
    print("Filled NaNs for lag and rolling mean features.")


    # Price vs Competitor (only if price and competitor_pricing are present)
    if 'price' in df.columns and 'competitor_pricing' in df.columns:
        df['price_vs_competitor'] = df['price'] - df['competitor_pricing']
        print("Successfully created 'price_vs_competitor' feature.")
    else:
        print("Warning: 'price' or 'competitor_pricing' not found, skipping 'price_vs_competitor' feature.")


    # --- Categorical Feature Encoding (for Prophet regressors) ---
    categorical_cols_to_encode = ['category', 'region', 'weather_condition', 'seasonality', 'promotion', 'epidemic', 'demand']

    for col in categorical_cols_to_encode:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')
            # Use a temporary DataFrame to avoid potential issues with chained assignment
            # or in-place modification warnings when using get_dummies
            df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, df_dummies], axis=1)
            # Drop original categorical column if it was successfully encoded
            df = df.drop(columns=[col])
            print(f"Successfully one-hot encoded and dropped original '{col}'.")
        elif col in df.columns and df[col].dtype != 'object':
            print(f"Warning: Column '{col}' is not of object type (it's {df[col].dtype}), skipping one-hot encoding.")
        elif col not in df.columns:
            print(f"Warning: Categorical column '{col}' not found in data. Skipping one-hot encoding for this column.")

    # Ensure all boolean/dummy columns are integers (0 or 1)
    for col in df.columns:
        if df[col].dtype == 'bool': # Check if column exists and is bool type
            df[col] = df[col].astype(int)
    print("Ensured boolean/dummy columns are integers.")

    print("Feature engineering complete. Final DataFrame columns:")
    print(df.columns.tolist())
    return df

def save_processed_data(df, engine):
    """
    Saves the engineered features to the 'processed_sales' table in PostgreSQL.
    """
    print(f"\n--- Saving processed data to '{PROCESSED_TABLE_NAME}' ---")
    if df.empty:
        print("No processed data to save. Exiting save operation.")
        return False

    try:
        # Dynamically determine column types based on actual DataFrame dtypes
        processed_column_types = {}
        for col in df.columns:
            if col == 'date':
                processed_column_types[col] = Date
            elif df[col].dtype == object: # String type (e.g., store_product_id, or original category/region if not dropped)
                processed_column_types[col] = String(255) # Use a generous string length
            elif df[col].dtype == 'int64' or df[col].dtype == 'int32':
                processed_column_types[col] = Integer
            elif df[col].dtype == 'float64':
                processed_column_types[col] = Numeric(10, 2) # Adjust precision/scale as needed
            elif df[col].dtype == 'bool': # Should have been converted to int already, but as a fallback
                processed_column_types[col] = Integer
            else:
                print(f"Warning: Unhandled dtype for column '{col}': {df[col].dtype}. Defaulting to String(255).")
                processed_column_types[col] = String(255)


        with engine.connect() as connection:
            connection.execute(text(f"DROP TABLE IF EXISTS {PROCESSED_TABLE_NAME} CASCADE;"))
            connection.commit()
            print(f"Existing table '{PROCESSED_TABLE_NAME}' dropped (if it existed).")

            # Use a larger chunksize for potentially faster insertion
            df.to_sql(PROCESSED_TABLE_NAME, engine, if_exists='append', index=False, dtype=processed_column_types)
            print(f"Processed data successfully saved to '{PROCESSED_TABLE_NAME}' table in PostgreSQL.")
            return True
    except Exception as e:
        print(f"Error saving processed data: {e}")
        return False

if __name__ == "__main__":
    print("--- Running 02_feature_engineering.py - Version 2025-07-30_Fix_B ---") # New version identifier
    engine = None
    try:
        engine = get_db_engine()
        if engine is None:
            print("Could not establish database engine connection. Exiting pipeline.")
        else:
            cleaned_data_df = load_cleaned_data(engine)

            if cleaned_data_df is not None and not cleaned_data_df.empty:
                processed_data_df = engineer_features(cleaned_data_df)
                if not processed_data_df.empty:
                    save_processed_data(processed_data_df, engine)
                else:
                    print("Feature engineering resulted in empty DataFrame. Skipping save (due to a previous critical error/return).")
            else:
                print("Cleaned data not loaded or is empty. Cannot proceed with feature engineering.")

    finally:
        print("\nFeature engineering pipeline completed.")