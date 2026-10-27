# Bike-Sharing Demand Forecasting Using Machine Learning and PCA

## Project Overview

This project predicts hourly bike rental demand using machine learning regression models. It analyzes historical rental patterns, weather conditions, and time-related features to estimate the number of bike rentals per hour.

The project explores baseline regression, Random Forest, Gradient Boosting, and Principal Component Analysis (PCA) to understand how different modeling approaches affect prediction performance.

## Objectives

- Analyze historical bike rental demand.
- Explore the effects of time, working days, and weather on rentals.
- Build and compare machine learning regression models.
- Apply PCA for dimensionality reduction.
- Evaluate predictions using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- Analyze feature importance and prediction errors.
- Save and reload the trained model for reuse.

## Dataset

**Source:** [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

This project uses the hourly dataset, `hour.csv`.

- **Observations:** 17,379
- **Columns:** 17
- **Period:** 2011–2012
- **Prediction target:** `cnt`, the total number of bike rentals per hour

### Important Features

| Feature | Description |
|---|---|
| `season` | Season of the year |
| `yr` | Year indicator |
| `mnth` | Month |
| `hr` | Hour of the day |
| `holiday` | Holiday indicator |
| `weekday` | Day of the week |
| `workingday` | Working-day indicator |
| `weathersit` | Weather category |
| `temp` | Normalized temperature |
| `atemp` | Normalized feels-like temperature |
| `hum` | Normalized humidity |
| `windspeed` | Normalized wind speed |
| `cnt` | Total hourly bike rentals |

The `casual` and `registered` columns were excluded from the model inputs because they directly contribute to the target `cnt`, causing target leakage.

The `instant` column was also excluded because it is a record identifier.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

## Project Workflow

1. Loaded and inspected the hourly dataset.
2. Checked data types and missing values.
3. Analyzed the distribution of hourly rentals.
4. Explored demand by hour, working-day status, and weather.
5. Converted the date column to datetime and sorted records chronologically.
6. Selected 12 input features and the target variable.
7. Created a chronological train-test split using complete dates.
8. Established a constant-mean baseline.
9. Trained and evaluated Random Forest and Gradient Boosting regressors.
10. Visualized actual versus predicted rentals.
11. Analyzed Random Forest feature importance.
12. Examined Gradient Boosting prediction errors.
13. Applied StandardScaler and PCA before Gradient Boosting.
14. Compared models with and without PCA.
15. Saved the trained Gradient Boosting model and ordered feature list.
16. Reloaded the saved model and tested a sample prediction.

## Exploratory Data Analysis

### 1. Hourly Demand Patterns

Bike rental demand varies considerably throughout the day.

The exploratory analysis showed prominent demand peaks around morning commuting hours and late afternoon to early evening. The highest average demand in the hourly analysis occurred around 5 PM.

### 2. Working-Day Analysis

Average hourly rentals were approximately:

- **Non-working days:** 181
- **Working days:** 193

The dataset showed slightly higher average hourly demand on working days. These averages do not capture every difference between weekday and weekend hourly patterns.

### 3. Weather Analysis

Average hourly rentals by weather category:

| Weather Category | Average Rentals |
|---|---:|
| Clear / Partly Cloudy | 204.87 |
| Mist / Cloudy | 175.17 |
| Light Rain / Snow | 111.58 |
| Heavy Rain / Snow | 74.33 |

The observed averages decreased as weather conditions became more severe.

## Machine Learning Models

### 1. Baseline Model

A simple baseline predicts the mean rental count from the training data for every test observation.

This establishes a reference point for evaluating the machine learning models.

### 2. Random Forest Regressor

Random Forest combines multiple decision trees to model nonlinear relationships between input features and rental demand.

Configuration used:

- `n_estimators = 100`
- `max_depth = 20`
- `min_samples_leaf = 2`
- `random_state = 42`

### 3. Gradient Boosting Regressor

The project uses `HistGradientBoostingRegressor`, which builds decision trees sequentially to improve predictions.

Configuration used:

- `max_iter = 200`
- `learning_rate = 0.1`
- `max_leaf_nodes = 31`
- `random_state = 42`

### 4. PCA + Gradient Boosting

Principal Component Analysis (PCA) transforms standardized input features into a smaller set of components while retaining as much variance as possible.

The PCA pipeline uses:

- `StandardScaler` for feature standardization.
- `PCA(n_components=0.95)` to retain at least 95% of the variance in the standardized training features.
- `HistGradientBoostingRegressor` for demand prediction.

In the earlier experiment, PCA reduced the 12 original features to 9 principal components and retained 95.24% of the variance.

## Model Performance

### Updated Gradient Boosting Evaluation

The Gradient Boosting model was retrained and evaluated using a chronological, date-based split. Complete dates were assigned to either training or testing, avoiding overlap between the two sets.

| Evaluation Detail | Result |
|---|---:|
| Total observations | 17,379 |
| Training observations | 13,891 |
| Testing observations | 3,488 |
| Input features | 12 |
| Mean Absolute Error (MAE) | 42.28 |
| Root Mean Squared Error (RMSE) | 65.04 |

**Current model:** `HistGradientBoostingRegressor`

**Interpretation:**

- **MAE = 42.28:** The model's predictions differed from actual demand by an average absolute amount of approximately 42.28 rentals per hour across the test set.
- **RMSE = 65.04:** This metric gives greater weight to larger prediction errors.

Lower MAE and RMSE values indicate better predictive performance when comparing models evaluated on the same test observations.

### Earlier Model Comparison

The following results were obtained in an earlier experiment using the previous train-test split. They have not yet been re-evaluated using the corrected date-based split and should not be directly compared with the updated Gradient Boosting results.

| Model | Input Representation | Earlier MAE | Earlier RMSE |
|---|---|---:|---:|
| Baseline | Training-set mean | 174.98 | 232.61 |
| Random Forest | 12 original features | 45.00 | 69.82 |
| Gradient Boosting | 12 original features | 40.52 | 62.95 |
| PCA + Gradient Boosting | 9 principal components | 113.22 | 165.90 |

The earlier experiment suggested that Gradient Boosting using the original features performed better than the other tested approaches. A fair comparison requires retraining and evaluating all models on the same corrected date-based split.

## PCA Experiment Findings

The earlier PCA experiment reduced the feature count from 12 to 9 while retaining 95.24% of the variance in the standardized training features.

However, the PCA-based model had higher errors than the Gradient Boosting model using the original features in that experiment.

This demonstrates an important machine learning concept: **preserving feature variance does not necessarily preserve the information most useful for predicting a target variable.**

PCA is an unsupervised dimensionality reduction technique. It selects components based on feature variance rather than their direct relationship with bike rental demand.

The PCA and non-PCA models should be compared again using the same corrected train-test split before drawing final conclusions about their relative performance.

## Prediction Error Analysis

The project uses the following evaluation and analysis methods:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Actual-versus-predicted visualizations
- Prediction error distribution
- Feature importance analysis

The updated Gradient Boosting model achieved a test MAE of 42.28 and a test RMSE of 65.04.

A sample prediction using the newly saved model produced:

- **Actual rentals:** 47
- **Predicted rentals:** 48.64
- **Absolute error:** 1.64

This individual prediction was close to the actual value. However, a single prediction is illustrative and does not represent the model's typical error. Overall test metrics provide a more reliable summary of performance.

## Model Saving and Reuse

The updated Gradient Boosting model and its ordered feature list were saved using Joblib.

Saved files:

- `models/gradient_boosting_bike_demand.pkl`
- `models/bike_demand_features.pkl`

The updated model was successfully reloaded and tested on a sample from the test dataset.

The saved feature list preserves the expected input column order when making predictions.

## Project Structure

```text
bike-sharing-demand-forecasting/
├── bike_sharing_demand_forecasting.ipynb
├── hour.csv
├── README.md
└── models/
    ├── gradient_boosting_bike_demand.pkl
    └── bike_demand_features.pkl
```

## Limitations and Future Improvements

- Re-evaluate the baseline, Random Forest, and PCA models using the corrected date-based split.
- Perform time-aware cross-validation and hyperparameter tuning using training data only.
- Compare PCA and non-PCA models using the same evaluation strategy.
- Investigate prediction errors during peak-demand periods.
- Explore time-based and lagged-demand features, ensuring that only information available at prediction time is used.
- Build a prediction interface for easier interaction.
- Evaluate model stability on later periods before considering deployment.

The reported updated metrics come from a chronological holdout experiment. Further validation would be needed before deploying the model in a real bike-sharing operation.

## Conclusion

This project demonstrates an end-to-end machine learning workflow for hourly bike rental demand forecasting, including exploratory data analysis, feature selection, chronological data splitting, regression modeling, PCA-based dimensionality reduction, prediction error analysis, and model persistence.

The updated Gradient Boosting model achieved an MAE of 42.28 and an RMSE of 65.04 on the corrected test split. The model was also saved, reloaded, and successfully used to generate a sample prediction.

The project highlights the importance of time-aware evaluation and demonstrates that dimensionality reduction may simplify input features without necessarily improving predictive performance.

## Author

**Siva Bharathi**

GitHub: [bharathi14siva-hub](https://github.com/bharathi14siva-hub)
