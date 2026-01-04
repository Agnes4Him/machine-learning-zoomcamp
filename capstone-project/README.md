## Projects idea
* Sales forecasting - Regression

* Employee Attrition prediction - Classification

* Energy Consumption Prediction - Regression  (use)

* Sentiment analysis on healthcare dataset     (use)

* DataTalks Club FAQ upload text classification task  (For later)

## Dataset for energy consumption prediction
https://www.kaggle.com/datasets/govindaramsriram/energy-consumption-dataset-linear-regression      ***

https://www.kaggle.com/datasets/mrsimple07/energy-consumption-prediction

https://www.kaggle.com/datasets/mexwell/smart-home-energy-consumption      ***

## Dataset for healthcare sentiment analysis
https://www.kaggle.com/datasets/thedevastator/german-2021-patient-reviews-and-ratings-of-docto

https://www.kaggle.com/datasets/junaid6731/hospital-reviews-dataset

https://archive.ics.uci.edu/dataset/461/drug+review+dataset+druglib+com     // Drug review

## MLFlow commands
```bash
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000 \
--cors-allowed-origins "*" \
--x-frame-options NONE \
--disable-security-middleware
```

NB - This avoid the MLFlow error `mlflow Invalid Host header - possible DNS rebinding attack detected`

### Run ML Pipeline

* uv run python train_pipeline/pipeline.py 

* uv run prefect deployment run 'training-flow/energy-consumption-training-pipeline'      *** To trigger immediate run

### Run FastAPI server
uv run python server/predict.py

### Run tests
pytest tests/test_main.py         

OR

pytest

### Run MLFlow and Prefect in EC2
* Install UV
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # OR wget

wget -qO- https://astral.sh/uv/install.sh | sh
```

* Create folder - `ML-project`

* Run 
```bash
uv init

uv add mlflow prefect pandas scikit-learn numpy
```

* Create file `pipeline.py` in the folder and add code snippet

* Run
```bash
cd ML-project

uv run mlflow server --cors-allowed-origins "*"

uv run mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000 \
--cors-allowed-origins "*" \
--x-frame-options NONE \
--disable-security-middleware
```

* Run
```bash
uv run python pipeline.py 

uv run prefect deployment run 'training-flow/energy-consumption-training-pipeline'
```
