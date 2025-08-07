import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.types import DateTime, Numeric, String
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import logging
import os
import warnings

# For interactive plots
import plotly.graph_objects as go
import plotly.io as pio

# Suppress Prophet's default logging to avoid clutter
logging.getLogger('prophet').setLevel(logging.WARNING)

# Suppress ALL Pandas PerformanceWarning messages
warnings.filterwarnings(
    'ignore', 
    category=pd.errors.PerformanceWarning
)

# --- Configuration (Centralized) ---
# Ensure this password matches your PostgreSQL setup exactly
DATABASE_URL = "postgresql://retail_user:salesforecast@localhost:5432/retail_sales_db"
PROCESSED_TABLE_NAME = 'processed_sales'
FORECAST_TABLE_NAME = 'sales_forecasts'
METRICS_TABLE_NAME = 'forecast_metrics'
TEST_SPLIT_DATE = '2023-01-01' # Date to split training and testing data
UNIQUE_ID_COL = 'store_product_id'
DATE_COL = 'date'
TARGET_COL = 'units_sold'

# Define columns that should NOT be treated as regressors in Prophet.
# This list includes identifiers, the target variable (both original and Prophet's 'y'),
# and original categorical columns that are expected to be replaced by their
# one-hot encoded versions in 'processed_sales'.
NON_REGRESSOR_COLS_BASE = [
    DATE_COL, UNIQUE_ID_COL, 'store_id', 'product_id', 
    # Original categorical columns that are now one-hot encoded and should be excluded
    'category', 'region', 'weather_condition', 'seasonality', 'promotion', 'epidemic', 'demand'
]

# Plotting Configuration
GENERATE_PLOTS = True
NUM_PLOTS_TO_GENERATE = 5 # Generate plots for the first N series
PLOTS_OUTPUT_DIR = 'output/forecast_plots'


# --- Database Engine ---
def get_db_engine():
    """Establishes and returns a database engine."""
    try:
        engine = create_engine(DATABASE_URL)
        # Test connection
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        print("Successfully connected to PostgreSQL via SQLAlchemy engine for forecasting.")
        return engine
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {e}")
        print("Please ensure PostgreSQL server is running and connection details are correct (host, port, user, password, database name).")
        print("Exiting forecasting pipeline.")
        return None

# --- Data Loading ---
def load_processed_data(engine):
    """Loads processed data from PostgreSQL."""
    print(f"--- Loading processed data from '{PROCESSED_TABLE_NAME}' ---")
    try:
        df = pd.read_sql_table(PROCESSED_TABLE_NAME, engine)
        print(f"Loaded {len(df)} rows from '{PROCESSED_TABLE_NAME}'.")

        # Ensure date column is datetime and set as index for sorting
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        
        # Ensure 'store_product_id' is treated as a string/object
        df[UNIQUE_ID_COL] = df[UNIQUE_ID_COL].astype(str)

        # Sort data by unique ID and date for correct processing
        df = df.sort_values(by=[UNIQUE_ID_COL, DATE_COL]).reset_index(drop=True)
        print("Data sorted by store_product_id and date.")
        
        return df
    except Exception as e:
        print(f"Error loading processed data: {e}")
        return None

# --- Evaluation Metrics ---
def evaluate_forecast_metrics(y_true, y_pred):
    """
    Calculates MAE, RMSE, and MAPE.
    Handles division by zero for MAPE by excluding actual zero values.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Calculate MAPE, handling division by zero for y_true (actual values)
    non_zero_true_indices = y_true != 0
    non_zero_true = y_true[non_zero_true_indices]
    non_zero_pred = y_pred[non_zero_true_indices]

    if len(non_zero_true) > 0:
        mape = np.mean(np.abs((non_zero_true - non_zero_pred) / non_zero_true)) * 100
    else:
        mape = np.nan # If all actual values are zero, MAPE is undefined

    return mae, rmse, mape

# --- Plotting Function ---
def plot_forecast(series_id, train_df, test_df, forecast_df, output_dir):
    """
    Generates an interactive Plotly forecast plot for a single series.
    Saves the plot as an HTML file.
    """
    fig = go.Figure()

    # Actuals (Training)
    fig.add_trace(go.Scatter(x=train_df['ds'], y=train_df['y'],
                             mode='lines', name='Actual (Train)',
                             line=dict(color='blue', width=1)))

    # Actuals (Test)
    fig.add_trace(go.Scatter(x=test_df['ds'], y=test_df['y'],
                             mode='lines', name='Actual (Test)',
                             line=dict(color='darkblue', width=2)))

    # Forecast
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'],
                             mode='lines', name='Forecast',
                             line=dict(color='red', width=2, dash='dot')))

    # Confidence Interval
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'],
                             mode='lines',
                             line=dict(width=0),
                             name='Upper Bound',
                             showlegend=False))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'],
                             mode='lines',
                             line=dict(width=0),
                             fillcolor='rgba(255,0,0,0.1)',
                             fill='tonexty',
                             name='Confidence Interval',
                             showlegend=False))

    fig.update_layout(
        title=f'Sales Forecast for {series_id}',
        xaxis_title='Date',
        yaxis_title='Units Sold',
        hovermode='x unified',
        template='plotly_white'
    )

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'forecast_{series_id}.html')
    pio.write_html(fig, file=plot_path, auto_open=False)
    print(f"Saved plot for {series_id} to {plot_path}")


# --- Forecasting ---
def train_and_evaluate_prophet(processed_df, test_split_date):
    """
    Trains a Prophet model for each unique store_product_id,
    makes forecasts, and evaluates performance.
    """
    all_forecasts = pd.DataFrame()
    all_metrics = pd.DataFrame()

    unique_series_ids = processed_df[UNIQUE_ID_COL].unique()
    print(f"--- Starting Prophet Model Training and Forecasting for {len(unique_series_ids)} series ---")

    plots_generated_count = 0

    for i, series_id in enumerate(unique_series_ids):
        # Print progress less frequently
        if (i + 1) % 10 == 0 or i == 0 or i == len(unique_series_ids) - 1:
            print(f"Processing Series {i+1}/{len(unique_series_ids)}: {series_id}")
        
        series_df = processed_df[processed_df[UNIQUE_ID_COL] == series_id].copy()

        # Rename columns for Prophet
        # This creates 'ds' and 'y' columns
        series_df = series_df.rename(columns={DATE_COL: 'ds', TARGET_COL: 'y'})

        # Define columns to exclude from regressors, including 'y' (Prophet's target)
        # and the original TARGET_COL if it somehow persists
        cols_to_exclude_from_regressors = NON_REGRESSOR_COLS_BASE + ['y', TARGET_COL]

        # Define regressors: all numeric columns not in the exclusion list
        regressors = [
            col for col in series_df.columns 
            if col not in cols_to_exclude_from_regressors and pd.api.types.is_numeric_dtype(series_df[col])
        ]
        
        # Safeguard: Fill NaNs in regressors. This is crucial for Prophet.
        for reg in regressors:
            if series_df[reg].isnull().any():
                series_df[reg] = series_df[reg].fillna(0)
        
        # Debugging: Print regressors being used for current series (uncomment for verbose debug)
        # if i == 0: 
        #     print(f"DEBUG: Regressors being used for {series_id}: {regressors}")

        # Split data into training and testing sets
        train_df = series_df[series_df['ds'] < test_split_date].copy()
        test_df = series_df[series_df['ds'] >= test_split_date].copy()

        if train_df.empty:
            # print(f"Skipping {series_id}: No training data before {test_split_date}.")
            continue
        if test_df.empty:
            # print(f"Skipping {series_id}: No testing data on or after {test_split_date}.")
            continue

        # Initialize and fit Prophet model
        model = Prophet(
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False # Daily seasonality often not needed for daily data
        )

        # Add regressors
        for regressor in regressors:
            model.add_regressor(regressor)

        try:
            model.fit(train_df)
        except Exception as e:
            print(f"Error fitting Prophet model for {series_id}: {e}")
            continue # Skip to next series

        # Make predictions for the test period
        future_df = test_df[['ds']].copy()
        
        # Merge the regressors from test_df into future_df for prediction
        # Ensure we only merge columns that are actual regressors
        future_df = future_df.merge(test_df[['ds'] + regressors], on='ds', how='left')
        
        # Ensure no NaNs in regressors for prediction
        for reg in regressors:
            if future_df[reg].isnull().any():
                future_df[reg] = future_df[reg].fillna(0)

        try:
            forecast = model.predict(future_df)
        except Exception as e:
            print(f"Error predicting with Prophet model for {series_id}: {e}")
            continue

        # Evaluate the forecast
        y_true = test_df['y'].values
        y_pred = forecast['yhat'].values

        if len(y_true) == len(y_pred) and len(y_true) > 0:
            # Clip predictions to be non-negative if units sold cannot be negative
            y_pred[y_pred < 0] = 0
            
            mae, rmse, mape = evaluate_forecast_metrics(y_true, y_pred)
            
            series_metrics = pd.DataFrame([{
                UNIQUE_ID_COL: series_id,
                'mae': mae,
                'rmse': rmse,
                'mape': mape
            }])
            all_metrics = pd.concat([all_metrics, series_metrics], ignore_index=True)
        else:
            print(f"Warning: Mismatch in lengths or empty data for evaluation for {series_id}. y_true: {len(y_true)}, y_pred: {len(y_pred)}. Skipping metrics for this series.")

        # Store forecasts
        forecast['store_id'] = series_df['store_id'].iloc[0] 
        forecast['product_id'] = series_df['product_id'].iloc[0]
        forecast[UNIQUE_ID_COL] = series_id
        
        # Ensure y_true aligns with forecast length (it should if len(y_true) == len(y_pred))
        forecast['y_true'] = test_df['y'].values if not test_df.empty and len(test_df) == len(forecast) else np.nan 
        
        # Ensure 'yhat' and other forecast columns are non-negative
        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
            if col in forecast.columns:
                forecast[col] = forecast[col].clip(lower=0)

        # Select relevant columns for saving
        forecast_subset = forecast[['ds', UNIQUE_ID_COL, 'y_true', 'yhat', 'yhat_lower', 'yhat_upper', 'store_id', 'product_id']]
        all_forecasts = pd.concat([all_forecasts, forecast_subset], ignore_index=True)

        # Generate plot for a sample of series
        if GENERATE_PLOTS and plots_generated_count < NUM_PLOTS_TO_GENERATE:
            plot_forecast(series_id, train_df, test_df, forecast, PLOTS_OUTPUT_DIR)
            plots_generated_count += 1


    return all_forecasts, all_metrics

def display_overall_results(all_metrics):
    """Displays aggregated evaluation metrics."""
    if not all_metrics.empty:
        print("\n--- Overall Forecast Evaluation Metrics ---")
        print(f"Average MAE: {all_metrics['mae'].mean():.2f}")
        print(f"Average RMSE: {all_metrics['rmse'].mean():.2f}")
        valid_mape = all_metrics['mape'].dropna()
        if not valid_mape.empty:
            print(f"Average MAPE: {valid_mape.mean():.2f}%")
        else:
            print("Average MAPE: N/A (no valid MAPE values to average)")
        print(f"Median MAE: {all_metrics['mae'].median():.2f}")
        print(f"Median RMSE: {all_metrics['rmse'].median():.2f}")
        if not valid_mape.empty:
            print(f"Median MAPE: {valid_mape.median():.2f}%")
        else:
            print("Median MAPE: N/A (no valid MAPE values to average)")
        print("------------------------------------------")
    else:
        print("\nNo metrics to display. Forecasting may have failed or no test data available for evaluation.")

# --- Data Saving ---
def save_forecasts_to_db(forecasts_df, engine):
    """Saves generated forecasts to PostgreSQL."""
    if forecasts_df.empty:
        print("No forecasts to save.")
        return

    print(f"Saving forecasts to '{FORECAST_TABLE_NAME}' table in PostgreSQL...")
    try:
        # Define column types for SQLAlchemy to ensure correct mapping
        forecast_column_types = {
            'ds': DateTime(), 
            UNIQUE_ID_COL: String(255),
            'y_true': Numeric(10, 2), 
            'yhat': Numeric(10, 2),
            'yhat_lower': Numeric(10, 2),
            'yhat_upper': Numeric(10, 2),
            'store_id': String(255),
            'product_id': String(255)
        }

        # Drop the table if it exists to ensure a clean insert
        with engine.connect() as connection:
            connection.execute(text(f"DROP TABLE IF EXISTS {FORECAST_TABLE_NAME} CASCADE"))
            connection.commit()
        print(f"Existing table '{FORECAST_TABLE_NAME}' dropped (if it existed).")

        forecasts_df.to_sql(
            FORECAST_TABLE_NAME,
            engine,
            if_exists='append',
            index=False,
            dtype=forecast_column_types,
            method=None
        )
        print(f"Forecasts successfully saved to '{FORECAST_TABLE_NAME}' table in PostgreSQL.")
    except Exception as e:
        print(f"Error saving forecasts to DB: {e}")

def save_metrics_to_db(metrics_df, engine):
    """Saves evaluation metrics to PostgreSQL."""
    if metrics_df.empty:
        print("No metrics to save.")
        return

    print(f"Saving metrics to '{METRICS_TABLE_NAME}' table in PostgreSQL...")
    try:
        metrics_column_types = {
            UNIQUE_ID_COL: String(255),
            'mae': Numeric(10, 2),
            'rmse': Numeric(10, 2),
            'mape': Numeric(10, 2)
        }
        
        with engine.connect() as connection:
            connection.execute(text(f"DROP TABLE IF EXISTS {METRICS_TABLE_NAME} CASCADE"))
            connection.commit()
        print(f"Existing table '{METRICS_TABLE_NAME}' dropped (if it existed).")

        metrics_df.to_sql(
            METRICS_TABLE_NAME,
            engine,
            if_exists='append',
            index=False,
            dtype=metrics_column_types,
            method=None
        )
        print(f"Metrics successfully saved to '{METRICS_TABLE_NAME}' table in PostgreSQL.")
    except Exception as e:
        print(f"Error saving metrics to DB: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Running 03_sales_forecasting.py - Version 2025-07-31_Plot_A ---")
    
    engine = get_db_engine()
    if engine is None:
        exit()

    processed_df = load_processed_data(engine)

    if processed_df is not None and not processed_df.empty:
        # Create output directory for plots if it doesn't exist
        os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
        print(f"Forecast plots will be saved to: {os.path.abspath(PLOTS_OUTPUT_DIR)}")

        all_forecasts, all_metrics = train_and_evaluate_prophet(processed_df, TEST_SPLIT_DATE)
        
        display_overall_results(all_metrics)
        
        save_forecasts_to_db(all_forecasts, engine)
        save_metrics_to_db(all_metrics, engine)
    else:
        print("Processed data not loaded or is empty. Cannot proceed with forecasting.")
    
    print("Forecasting pipeline completed.")
    print(f"\nCheck the '{PLOTS_OUTPUT_DIR}' directory for interactive forecast plots (if enabled and generated).")