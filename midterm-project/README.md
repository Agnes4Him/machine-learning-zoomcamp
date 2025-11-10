## Project idea
1. Breast lesion severity
https://archive.ics.uci.edu/dataset/161/mammographic+mass

2. Heart attack survival
https://archive.ics.uci.edu/dataset/38/echocardiogram

3. Medical Insurance cost
https://www.kaggle.com/datasets/mohankrishnathalla/medical-insurance-cost-prediction

https://www.kaggle.com/datasets/saadaliyaseen/decoding-medical-costs-analyzing-insurance-data

4. Fraud detection
https://www.kaggle.com/datasets/eshummalik/securepay-credit-card-fraud-detection-data

https://www.kaggle.com/datasets/darshandalvi12/fraud-detection-in-financial-transactions

5. Hotel reservation cancellation
https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset

6. Teens Phone Usage and Addiction
https://www.kaggle.com/datasets/sumedh1507/teen-phone-addiction


## Steps
* Download data

* Explore data

* Split dataset - train, val and test

* Extract target variable

* Delete target from dataset

* Train models using `train` dataset, and validate using `val` dataset

* Train using following - Linear regression, DecisionTreeRegressor, RandomForestRegressor, XGBoost + with tuning. Choose the best model

* Train the best model using full_train dataset

* Save model locally using pickle (eventually psuh to GitHub)

* Create web service using FastAPI to serve the model

* Test the web service

* Dockerize the web service

* Deploy with fly.io

* Documentation

uv run uvicorn predict:app --host 0.0.0.0 --port 8000 --reload