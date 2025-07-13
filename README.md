# Electric Power Consumption Time Series Forecasting

## Project Overview

This project analyzes and forecasts daily electricity consumption using time series data from a real household.
The goal is to compare the performance of several regression models—both classical machine learning and deep learning (LSTM)—and to evaluate the impact of rolling (moving average) feature engineering on prediction accuracy.

## Data Description

* **Source:** UCI Machine Learning Repository — "Individual household electric power consumption" dataset
* **Period:** Several years of minute-level data aggregated to daily totals
* **Target:** `Global_active_power` (total daily active power consumption)
* **Features:**

  * Calendar features: `month`, `dayofweek`, `season`
  * Rolling features: 3, 7, and 30-day moving averages of the target

## Workflow

1. **Data Preprocessing**

   * Load raw data, merge date & time, convert types, handle missing values
   * Aggregate to daily level
2. **Feature Engineering**

   * Extract calendar features, encode categoricals, add rolling (moving average) features (3, 7, 30 days)
3. **Train/Test Split**

   * Chronological split: 80% train, 20% test
4. **Modeling**

   * **Naive Forecast:** Yesterday’s value as today’s prediction
   * **Linear Regression**
   * **RandomForestRegressor**
   * **XGBRegressor**
   * **LSTM (Deep Learning):** Sequence-based model for time series forecasting
5. **Evaluation**

   * Metrics: MAE, RMSE, R², MAE\_RATIO (relative MAE)
   * Results reported before and after rolling feature engineering
   * Feature importance analysis for top models
6. **Visualization**

   * Bar charts comparing models and feature importances
7. **Saving Results and Model Artifacts**

   * Results saved as `.pkl` files for reproducibility
   * Best model (RandomForestRegressor) checkpoint saved

## Key Results

* **Classical ML Models:**
  All machine learning models outperformed the naive baseline.
  Random Forest achieved the lowest MAE and best MAE\_RATIO.

* **After Rolling Features:**
  Minor improvement in Random Forest, with small differences between all models.
  14-day rolling was tested but not beneficial, so dropped from final features.

* **LSTM Results:**
  Despite extensive tuning and early stopping, the LSTM model did **not outperform** classical ML models.
  Example of LSTM output:

  ```
  Early stopping at epoch 44
  MAE_LSTM: 1362.3
  MSE_LSTM: 2064982.25
  R2_LSTM: -8.88
  RMSE_LSTM: 1437.0
  ```

  The model showed high MAE and negative R², suggesting the data is either too noisy or not suited for sequence-based deep learning without further domain features.

* **Feature Importance:**
  Rolling means (especially 3 and 7 days), season, and day of week were the most important predictors.

## Lessons Learned & Conclusions

* Not all time series benefit from deep learning (LSTM); on this dataset, classical ML methods were consistently superior.
* Feature engineering (rolling means, calendar effects) is crucial, but not all moving windows improve results.
* Professional data science requires documenting both **successes and limitations**:
  All results, including unsuccessful LSTM experiments, are fully reported here.

## Project Pipeline Order

The recommended file/module order for running and understanding the project workflow is:

1. `data_preprocessing` – Data loading and cleaning
2. `feature_engineering` – Feature creation and encoding
3. `model_naive_forecast` – Baseline naive forecast model
4. `linear_regression` – Linear regression pipeline
5. `random_forest` – Random forest pipeline and feature importance
6. `model_xgboost` – XGBoost regression pipeline and feature importance
7. `visualization` – Model and results visualization utilities
8. `feature_engineering_lstm` – Specialized feature engineering for LSTM
9. `build_dataset` – Custom PyTorch dataset for LSTM
10. `dataloaders` – PyTorch DataLoader wrappers
11. `model_lstm` – LSTM model definition
12. `train_lstm` – LSTM model training loop
13. `evaluator` – Model evaluation metrics and utilities
14. `main` – Main script orchestrating the entire workflow

* `results_before_MA.pkl` / `results_after_MA.pkl`: Saved model metrics before/after rolling features
* `best_model.pkl`: Best model checkpoint (Random Forest)
* `requirements.txt`: Python dependencies

> **Tip:**  
> All files are modular and imported as needed by `main.py`.  
> To reproduce the full analysis, simply run `main.py`, which sequentially utilizes all components above.



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

**Marina Kurland**
*(This project was completed as part of my independent learning in data analysis and machine learning.)*

Special thanks to the UCI ML Repository for the dataset, and to the scikit-learn, XGBoost, and PyTorch teams for their frameworks.

---

**Contact:** For questions, suggestions, or improvements, please open an issue or contact via GitHub.

