# Electric Power Consumption Time Series Forecasting

## Project Overview

This project analyzes and forecasts daily electricity consumption using time series data from a household power consumption dataset. The goal is to compare the performance of several regression models—both statistical and machine learning based—on this real-world dataset and to evaluate the impact of rolling (moving average) feature engineering.

## Data Description

* **Source:** UCI Machine Learning Repository, "Individual household electric power consumption" dataset
* **Period:** Several years of minute-level data aggregated to daily totals
* **Target:** `Global_active_power` (total active power consumption per day)
* **Features:**

  * Calendar features: `month`, `dayofweek`, `season`
  * Rolling features: 3, 7, and 30-day moving averages of the target

## Workflow

1. **Data Preprocessing**

   * Loading the raw data
   * Combining date and time, converting types, handling missing values
   * Aggregating to daily level
2. **Feature Engineering**

   * Calendar-based features (month, weekday, season)
   * Encoding categorical features
   * Adding rolling (moving average) features (3, 7, 30 days)
3. **Train/Test Split**

   * Chronological split (80% train, 20% test)
4. **Modeling**

   * **Naive Forecast**: Uses yesterday’s value as today’s prediction
   * **Linear Regression**
   * **RandomForestRegressor**
   * **XGBRegressor**
5. **Evaluation**

   * Metrics: MAE, RMSE, MAE\_RATIO (relative MAE)
   * Baseline results (before rolling features) and improved results (after)
   * Feature importance analysis for the best models
6. **Visualization**

   * Bar charts comparing models and feature importances
7. **Saving Results and Model Artifacts**

   * Results are saved to `.pkl` files for reproducibility
   * Best model (RandomForestRegressor) is saved for future use

## Key Results

* **Baseline (No Rolling):** All ML models outperformed the naive baseline. Random Forest showed the best results.
* **After Rolling Features:** Slight improvement in Random Forest, but differences between models remained small. Rolling window of 14 days was not beneficial and was dropped.
* **Best Model:** RandomForestRegressor (lowest MAE and MAE\_RATIO)
* **Feature Importance:** Rolling means (especially 3 and 7 days), season, and day of week were most important for prediction.

## File Structure

* `data_preprocessing.py`: Data loading and cleaning functions
* `feature_engineering.py`: Feature extraction and encoding
* `model_naive_forecast.py`: Naive forecast baseline
* `linear_regression.py`: Linear regression pipeline
* `random_forest.py`: Random forest pipeline and feature importance
* `model_xgboost.py`: XGB pipeline and feature importance
* `visualization.py`: All project plots and reporting utilities
* `main.py`: Main script to run full workflow
* `results_before_MA.pkl`/`results_after_MA.pkl`: Saved model metrics before/after rolling features
* `random_forest_best.pkl`: Best model checkpoint
* `requirements.txt`: List of Python dependencies

## How to Run

1. Clone the repository and install requirements:

   ```bash
   pip install -r requirements.txt
   ```
2. Place the raw data file (`household_power_consumption.txt`) in the project directory.
3. Run the main pipeline:

   ```bash
   python main.py
   ```
4. Results and artifacts will be generated in the working directory.

## Author

Author: Marina Kurland
*(This project was completed as part of my independent learning in data analysis and machine learning.)*

Special thanks to the UCI ML Repository for the dataset and the scikit-learn & XGBoost teams for the ML frameworks.

---

**Contact:** For questions, suggestions or improvements, please open an issue or contact via GitHub.
