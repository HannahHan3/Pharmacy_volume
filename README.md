# Pharmacy_volume
## Phase 1:

Data are provided by a technology company that support data-driven healthcare. The data is time series data about pharmacy sales volume. The goal is to provide insights about factors that have great influence on pharmacy sales and deployable model that could be used to predict pharmacy sales.

I considered time series method like ARIMA, and also basic ML models, for which I utilized last 7-days volume as new features according to the EDA results.

For more details, please see the "Phase 1 Demo.doc"
## Phase 2:










We got historical data about pharmacy sales volume, and we need to predict the next 35-70 days' drug volume in advance.

Since I've signed confidentiality agreement, the data are not provided here. Yet I will show the model deployment files with sample input. I also attached sample code file for modeling part for reference.

For modeling part, I've tried GDBoosting, Bagging, Random Forest, SGDBoosting, Lasso, Ridge. After parameter tuning, i dumped the best models into .sav files for deployment.
