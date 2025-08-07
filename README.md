# Sales-Forecast-and-Demand-Planning

## Project Overview

This project provides a robust, end-to-end solution for retail sales forecasting and demand planning.The dataset is sourced from Kaggle. By integrating a modular Python pipeline for data processing and machine learning, a PostgreSQL database for scalable data storage, and an interactive Power BI dashboard for business intelligence, this system delivers accurate sales forecasts and actionable insights. The goal is to enable data-driven decisions that optimize inventory management, enhance supply chain efficiency, and ultimately boost profitability.

-----

## 1\. Project Pipeline & Python Scripts Analysis

The project workflow is structured into a logical, three-step Python pipeline. Each step is encapsulated in a separate script, ensuring modularity and reusability.

### **1.1. Data Acquisition and Preprocessing (`01_ingest_and_clean_data.py`)**

This script is the entry point of the pipeline. It focuses on ingesting raw data and performing essential cleaning to ensure data quality and integrity before any further analysis.

  * **Process:** The script loads the raw `sales_data.csv` file. It then performs a series of cleaning operations, including handling data type conversions and validating key columns. The cleaned data is then successfully exported to a PostgreSQL database, a crucial step for establishing a reliable data source.
  * **Key Insight:** This initial step highlights the importance of a **centralized data environment**. By moving the cleaned data into a PostgreSQL table (`cleaned_sales`), the project creates a single, version-controlled source of truth. This prevents discrepancies and ensures that all subsequent analyses and models work with the exact same dataset. The use of SQLAlchemy and psycopg2 demonstrates a professional approach to database connectivity in Python.

### **1.2. Feature Engineering (`02_feature_engineering.py`)**

This script takes the preprocessed data and transforms it into a richer format by creating new features that are highly valuable for time-series forecasting.

  * **Process:** The script retrieves the cleaned data from the `cleaned_sales` table in PostgreSQL. It then engineers several new features, including:
      * **Date-related Features:** Extracting the month, week, and day of the week to capture seasonal and weekly sales cycles.
      * **Lagged Features:** Creating a lagged sales column to represent the sales of the previous period. This is essential for a time-series model to learn from historical patterns.
      * **Rolling Averages:** Computing rolling averages to smooth out short-term fluctuations and identify underlying trends.
  * **Key Insight:** This step demonstrates the value of **creating predictive signals**. The engineered features provide the forecasting model with more context about the data, significantly improving its ability to recognize patterns and make more accurate predictions. The script's output, a new table named `processed_sales` in PostgreSQL, ensures that the feature engineering logic is reproducible and the enhanced data is readily available for the next stage.

### **1.3. Sales Forecasting and Model Evaluation (`03_sales_forecasting.py`)**

This is the core machine learning component, where the time-series model is trained and evaluated.

  * **Process:** The script loads the processed data from PostgreSQL, trains a **Prophet model** (a forecasting library developed by Facebook), and then generates future sales forecasts. The model is trained on a portion of the data and evaluated on a separate test set to measure its performance. Finally, both the raw forecasts and key performance metrics (e.g., MAE, MSE) are saved back to the PostgreSQL database in dedicated tables (`sales_forecasts` and `forecast_metrics`).
  * **Key Insight:** The use of a **Prophet model** is particularly insightful as it is designed to handle seasonality and holidays automatically, making it well-suited for retail data. The script's methodology of saving both the predictions and the evaluation metrics to the database is a best practice for **ModelOps (MLOps)**. It allows for continuous monitoring of the model's accuracy over time and provides a historical record for comparing different model versions.

-----

## 2\. PostgreSQL Database

The PostgreSQL database is the backbone of this project, ensuring data integrity, scalability, and a unified environment for all components.

  * **Role:** It acts as the central data repository, storing the cleaned data, processed features, and final model outputs.
  * **Key Tables:**
      * `cleaned_sales`: The initial, cleaned dataset.
      * `processed_sales`: The dataset with engineered features.
      * `sales_forecasts`: The table containing the final sales predictions.
      * `forecast_metrics`: A table to track the performance of the forecasting model over time.

-----

## 3\. Power BI Dashboard Insights

The Power BI dashboard provides a dynamic and intuitive interface for business users to explore sales trends and forecast data without needing to write any code. It is directly connected to the PostgreSQL database.

Executive Summary (Page 1)
This page provides a high-level overview of sales and forecast performance.
•	Overall Insight: The page effectively communicates your key metrics at a glance. The clear separation of actual sales, forecasted sales, and overall forecast accuracy is a great way to start the dashboard. The Average MAPE of 41.80% suggests there's a significant opportunity to improve your forecast model's accuracy.
•	Visual-Specific Insights:
o	KPI Cards: The use of three distinct KPI cards is a great way to present key metrics. The color-coding helps differentiate them quickly.
o	Line Chart (Aggregated Sales & Forecast Over Time): This chart shows a clear downward trend in both actual and forecasted sales. This is a critical trend that should be investigated further.
o	Bar Chart (Average MAPE by Category): The red color on the bars suggests that the forecast error is high across all categories. You should consider sorting this chart to immediately identify the categories with the highest error.
________________________________________
Forecast Deep Dive (Page 2)
This page is designed for a detailed analysis of a single store_product_id series.
•	Overall Insight: This page is well-designed for a drill-down analysis. The combination of key error metrics (MAE, MAPE, RMSE), a line chart, and a data table provides a complete view of a single series. The shaded confidence interval on the line chart is a great way to visualize the uncertainty of your forecast.
•	Visual-Specific Insights:
o	KPI Cards: The individual MAE, MAPE, and RMSE cards for the selected series are very effective. The values of 2.81K (MAE), 4.18K (MAPE), and 3.56K (RMSE) show the magnitude of your forecast errors for the selected series.
o	Line Chart (Selected Series): The chart shows a good fit between the actual and forecasted sales. However, the forecast line seems to consistently slightly under-predict the actuals in some areas. This might indicate a slight bias in your model.
________________________________________
Forecast Accuracy Analysis (Page 3)
This page is intended for analyzing forecast accuracy across different stores and products.
•	Overall Insight: This page provides a solid foundation for understanding the drivers of your forecast error. The combination of a bar chart and a scatter plot allows you to analyze your forecast accuracy from multiple angles.
•	Visual-Specific Insights:
o	Bar Chart (Average MAPE by Store ID): This chart shows that some stores have a much higher MAPE than others. Sorting this chart in descending order by MAPE would make it much easier to identify the stores that need the most attention.
o	Scatter Plot (Actuals vs. Forecasts): The data points on this chart show a strong positive correlation between actual and forecasted sales. However, the points are not perfectly aligned with a 1:1 trend line, indicating that the model has some level of error. Most of the points seem to be slightly above the 1:1 trend line, which suggests a slight tendency for your model to under-forecast.
o	Data Table: The table at the bottom provides a detailed view of the forecast accuracy for different categories and products. This is a great way to supplement the visuals and get a deeper understanding of the data.
________________________________________
Forecast Drivers Analysis (Page 4)
This page is dedicated to a more in-depth look at what contributes to forecast error.
•	Overall Insight: This page is well-designed to help you understand the root causes of your forecast errors. The mix of visuals provides a comprehensive view of how forecast accuracy varies across different stores, categories, and their contribution to the total error. The Average MAPE card at 41.80% again highlights the overall need for model improvement.
•	Visual-Specific Insights:
o	Clustered Column Chart (Error by Store vs. Product Category): This visual shows a clear distribution of error across different stores and product categories. It reveals which store-category combinations have the highest errors. The large red columns for store_id S001 and S002 indicate that these stores have particularly high errors.
o	Funnel Chart (Error Distribution by Product Category): This chart shows that the forecast errors are distributed fairly evenly across categories. The slight variations can be a starting point for a deeper dive into a specific category.
o	Waterfall Chart (Contribution of Error by Store): This chart clearly highlights the stores that contribute the most to the total forecast error. The large, green columns indicate which stores have the highest error contribution. This is an excellent visual for prioritizing which stores to investigate first.

Here are the key business questions that your dashboard, as a whole, can answer, broken down by page and visualization.

