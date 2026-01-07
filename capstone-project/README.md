# ENERGY CONSUMPTION PREDICTION MACHINE LEARNING CAPSTONE PROJECT

## Problem Statement

Energy consumption forecasting is a fundamental problem in modern energy management systems, as it directly supports efficient resource allocation, cost reduction, and sustainability initiatives. With increasing energy demand and variability in household usage patterns, traditional rule-based or aggregate forecasting approaches are often insufficient to capture the complex relationships between time, environmental conditions, appliance usage, and household characteristics.

This project focuses on developing a machine learning–based predictive model to estimate household energy consumption using historical data. The dataset used was sourced from Kaggle and contains approximately 100,000 records of household energy usage. Each record includes attributes such as home_id, appliance_type, historical energy_consumption_kwh, timestamp information (date and time), outdoor_temperature, season, and household_size. The target variable for the prediction task is energy consumption.

A significant challenge in this problem is effectively modelling the temporal dynamics of energy usage. Energy consumption varies not only across households and appliances, but also across different times of day, days of the week, and seasons. To address this challenge, feature engineering was applied to extract granular temporal features from the raw timestamp data, including day, day of week, hour, and minute. These engineered features enable the models to learn cyclical and behavioural consumption patterns more effectively.

Several supervised machine learning models were trained and evaluated, including Decision Tree, Random Forest, and XGBoost. Comparative evaluation showed that the Random Forest model achieved the best predictive performance, indicating its effectiveness in capturing non-linear relationships and interactions between features in the dataset.

The final model is intended to serve as a predictive tool for estimating household energy consumption based on historical, environmental, and temporal factors. This solution can be used to support energy demand forecasting, identify high-consumption patterns, and inform decision-making in energy management and optimisation systems.

## Project Objectives

The key objectives of this project are:

To analyse household energy consumption data and identify key factors influencing energy usage.

To apply feature engineering techniques to improve representation of temporal and contextual information.

To train and compare multiple machine learning models for energy consumption prediction.

To evaluate model performance and select the most effective predictive model.

To develop a reusable prediction model that can support energy planning and management use cases.

To deploy a web service to serve the model, using best practices

## Scope and Limitations

1. Scope

The project focuses on supervised machine learning approaches for regression-based energy consumption prediction.

Predictions are based on historical consumption, household attributes, appliance type, environmental conditions, and engineered time features.

Model evaluation is performed using standard performance metrics to compare multiple algorithms.

2. Limitations

The dataset is sourced from Kaggle and may not fully represent real-world energy usage across all geographic regions or household types.

External factors such as energy pricing, occupancy behaviour changes, or unexpected events are not included in the dataset.

The model’s performance is dependent on the quality and completeness of historical data and may require retraining for deployment in different contexts.

##############################################################################################################################################

## Model Training

Model training was achieved by following the following steps:

### Data Loading

The dataset for the task was sourced from Kaggle, and loaded into Jupyter notebook (`./capstone.ipynb`)using pandas library. There were 100,000 records total.

### Exploratory Data Analysis

To understand the dataset better, I did some analysis to find out:
- null values

- duplicate records

- existing columns and their data types

- description of the numerical columns

- values count for categorical columns

- distribution of the target variable - skewed to the right


### Data Cleaning

The dataset required only very minimal cleaning as there were no null or duplicate values. The columns were renamed to remove spacing and special characters.

### Feature Engineering

The date and time columns were converted to the `datetime` data type, and then combined to give a singular timestamp.

Both columns were merged into a `timestamp` column which was subsequently splitted into - `hour`, `minute`, `day_of_week`, `day`, `month`, `is_weekend`.

Feature importance was also done to determine the features to include in model training.

### Train Multiple Models

Multiple models were trained -

* Decision Trees

* RandomForest

* XGBoost

RandomForest gave the best Root Mean Squared Error - `RMSE`

### Experiment Tracking

Experiment tracking was done using `mlflow` which was installed using `UV`. Each of the 3 model above and their best paramenters, was trained within an experiment

and their parameters, `RMSE` scores, and artifact (DV and model) were logged to MLFlow server. MLFlow was later ran in Docker during deployment, to achieve

portability and reusability.

### Training Pipeline

The notebook used for the initial models training and experimentation was converted to a Python script and saved in `./train_pipeline/pipeline.py`.

`Prefect` was also used to orchestrate the training workflow.

To run the pipeline without docker:

* Navigate to the directory `capstone-project`, and run:

```bash
uv sync
```

* Start mlflow server with the command:

```bash
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000 \
--cors-allowed-origins "*" \
--x-frame-options NONE \
--disable-security-middleware
```

* Access MLFlow on `http://localhost:5000`

* Run the train pipeline with the commands:

```bash
uv run python train_pipeline/pipeline.py 

uv run prefect deployment run 'training-flow/energy-consumption-training-pipeline'       # On a different terminal to trigger the flow manually
```

* Visit `http://localhost:4200` to view the flow

To run the pipeline using docker:

* Build the docker image of the pipeline as follow:

```bash
cd train_pipeline

docker build -t <IMAGE_NAME>:<IMAGE_TAG> .
```

* Run MLFlow, Prefect server and training pipeline docker container

```bash
cd train_pipeline

docker-compose up
```

### Web Service
1. Steps to run

### Local Deployments - Kind
1. Steps to deploy and run

### Cloud Deployments - EKS
1. Steps to deploy and run

### CI/CD Pipeline - GitHub Actions

### Monitoring and Observability

### Future Improvements








### MLFlow commands
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

### CI pipeline components
✔ Run unit tests
✔ Generate code coverage
✔ Dependency vulnerability check
✔ SAST with SonarQube (external AWS server)
✔ Build Docker image
✔ Scan image with Trivy
✔ Push image to Docker Hub
✔ Update Kubernetes manifest with the new image tag


### Security considerations
Security Coverage Summary
Stage	Tool
Unit testing	pytest
Coverage	pytest-cov
Dependency scan	safety
SAST	SonarQube (external)
Container scan	Trivy
Image registry	Docker Hub
Manifest update	sed + git commit

### SonarQube properties file
sonar.projectKey=fastapi-web
sonar.projectName=fastapi-web
sonar.sources=server
sonar.tests=tests
sonar.python.coverage.reportPaths=coverage.xml
sonar.sourceEncoding=UTF-8

### Create K8s cluster with `kind`
```bash
kind create cluster --name ml --config capstone-project/kubernetes/local/kind/kind-config.yaml
```

### ArgoCD Docs
https://www.digitalocean.com/community/tutorials/how-to-deploy-to-kubernetes-using-argo-cd-and-gitops     ## Installations

https://argo-cd.readthedocs.io/en/stable/getting_started/       ## Creating Apps through UI

Commands
```bash
kubectl create namespace argocd

kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

watch kubectl get pods -n argocd

kubectl port-forward svc/argocd-server -n argocd 8080:443                    ## Port forwarding to access ArgoCD UI

kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d; echo            ## Retrieve ArgoCD password. User is `admin`
```

### Run SonarQube
```bash
sudo apt install docker.io

sudo usermod -aG docker $USER

newgrp docker 

docker run --name sonarqube -p 9000:9000 -d sonarqube:10.6-community
```

### Amazon EKS
