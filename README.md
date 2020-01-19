# Pharmacy_volume
## Phase 1: Initial modeling

Data are provided by a technology company that provide tools to support data-driven healthcare. The data is time series data about pharmacy sales volume. The goal is to provide insights about factors that have great influence on pharmacy sales and deployable model that could be used to predict pharmacy sales.

I considered time series method like ARIMA, and also basic ML models, for which I utilized last 7-days volume as new features according to the EDA results.

For more details, please see the "Phase 1 Demo.docx"

## Phase 2: Training 

Task: use the data in the past 35 days to predict the sales in the coming 35-70 days in advance.
Additional: Take whether in first 10 days or last 10 days of the month into account. Train models, conducted grid search to pick the best models and dumped as .sav files. For modeling part, I've tried GDBoosting, Bagging, Random Forest, SGDBoosting, Lasso, Ridge. For hyperprameter tuning, I manually did the "grid search".

Please see "Training.py" for more details.

## Phase 3: Deployment

Download the deployment fold, then run the .py file, you will see the how it work with sample input. 

