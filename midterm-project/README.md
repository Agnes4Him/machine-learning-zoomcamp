# INTRODUCTION
This is a midterm project for the `Machine Learning Zoomcamp - 2025` to demostrate my understanding being able to:

- Ientify a problem

- Describe the problem

- Determine if the problem can be solved using rule-base approach or Machine Learning

- Train models if using machine learning, and determine the best model following evaluation

- Use the model

# PROBLEM STATEMENT
### Title
Predicting Teens Smartphone Addiction Levels Using Behavioral, Lifestyle, and Mental Health Indicators

### Question
Can we predict a teen’s smartphone addiction level based on their daily phone use, demographics, sleep habits, mental health indicators, parental influence, and lifestyle factors?

### Problem Description
Smartphone use among teenagers has become nearly universal, offering educational and social benefits—but also potential risks related to addiction, sleep deprivation, and mental health issues such as self-esteem, depression, anxiety, social isolation, and so on. Excessive or uncontrolled smartphone use can negatively impact academic performance, relationships, and psychological well-being.

The goal of this project is to develop a predictive model that estimates the `Addiction Level` of teenagers using measurable behavioral, psychological, and lifestyle features such as:

- Demographics such as age and gender

- Daily usage hours (how long they use their phone each day)

- Usage purpose

- Sleep hours

- Time spent on social media / gaming / education

- Social interactions

- Academic performance

- Anxiety and Depression Levels

- Self-esteem

- Parental control

- Phone checks per day

- Number of Apps used

- Weekend usage

- Screen time before bed

By learning patterns in this data, the model can predict which teens are at higher risk of developing smartphone addiction and help inform preventive interventions, counseling programs, and awareness campaigns.

### Type of Problem
This is a `supervised regression` problem, since the target variable Addiction_Level is numeric (e.g., on a scale of 0–10).

Alternatively, it could be converted into a classification problem (e.g., Low, Moderate, High addiction risk).

### Target Variable
Addiction_Level

### How the Model Addresses the Problem

The model’s predictions could be used to:

* Detect early signs of smartphone addiction among teens.

* Personalize interventions: Schools or counselors can focus efforts on high-risk individuals.

* Inform policy: Evidence-based insights can guide screen time recommendations or parental control app designs.

* Educate teens and parents: Help them understand which behaviors (e.g., frequent night-time phone use) increase risk.

This predictive model serves as a decision-support tool, not a replacement for clinical judgment. It quantifies risk and helps prioritize attention toward teens who need support the most.

# IMPLEMENTATION STEPS
### Data Sourcing
The dataset used is the `teen_phone_addiction_dataset` found on Kaggle. This dataset was downloaded and slightly modified. The modified
version of the datase in CSV format used in this project can be found at [teen_phone_addiction_dataset](https://raw.githubusercontent.com/Agnes4Him/project-datasets/refs/heads/main/teen_phone_addiction_dataset2.csv).

## Exploratory Data Analysis
This revealed the following for `Addiction_Level`:
- count - 3000.000000
- mean  - 8.881900
- std   - 1.609598
- min   - 1.000000
- 25%   - 8.000000
- 50%   - 10.000000
- 75%   - 10.000000
- max   - 10.000000

There were no null or duplicate data in the dataset.

## Feature Engineering
An extra column name `Addiction_Category` was added to the dataset to categorize `Addiction_Level`.

## Model Training
This was achieved by splitting the data into train, validation and test datasets.
The following models were trained using train dataset, validated using validation dataset, and their `RMSE` was determined to find the best model for this use case:
- LinearRegressor

- RidgeRegressor

- LassoRegressor

- DecisionTreeRegressor

- RandomForestRegressor

- XGBoost

Different parameters were tested to find the best spot. XGBoost had the best performance.

XGBoost was retrained using a combination of train and validation dataset, and then tested using test dataset. This gave an `RMSE` of `0.659`.

The model was saved locally as `model.bin`using `pickle`.

The notebook used for the training `mid-term.ipynb` was converted to a Python script named `train.py`.

## Web Server
Uisng `FastAPI`, a web server was created to serve the saved model and make predictions on new data. Input validation and type checking was added.
for quality control. The setup was tested by sending a `POST` request from `test_predict.py`.

![Start Web Server](images/start-server-local.png)

![Server UI Local](images/server-ui-local.png)

## Deployment
Docker was used to containerize the application, and tested locally. The image created was pushed to Amazon ECR to be deployed to App Runner subsequently.




# Outline
* Introduction

* Problem statement

* Problem description

* Steps in execution
- Data source and download
- EDA
- Feature engineering
- Model training using different algorithms and evaluation. RMSE
- Full training
- Save model locally
- Convert notebook to script(`train.py`) and clean up
- Create a web server with FastAPI to run predictions (`predict.py`)
- Test API using `test_predict.py`
- Dockerize web server and push to ECR - show images of docker commands and ECR repo
- Deploy to App Runner - show images to confirm cloud deployment
- CI/CD pipeline for automatic update to web server

* How to reproduce/replicate
- Explain directory structure
- How to run web server locally and test prediction
- How to build and run docker image locally and test prediction
- How to push docker image to ECR
- How to deploy the image to App Runner and test prediction
- Using CI/CD pipeline to update the deployment

* Limitations and next steps 
- Experiment tracking
- Hyperparameter tuning
- Training pipeline
- Use of model registry


## Push Image to ECR

* Run:
```bash
aws ecr create-repository \
    --repository-name teen-addiction-prediction \
    --region us-east-1

```

* Authenticate:
```bash
aws ecr get-login-password --region us-east-1 | \
docker login --username AWS --password-stdin 759907441676.dkr.ecr.us-east-1.amazonaws.com
```

* Tag iamge:
```bash
docker tag teen-addiction-prediction:1.0.0 759907441676.dkr.ecr.us-east-1.amazonaws.com/teen-addiction-prediction:1.0.0
```

* Push image:
```bash
docker push 759907441676.dkr.ecr.us-east-1.amazonaws.com/teen-addiction-prediction:1.0.0
```

* Verify:
```bash
aws ecr list-images --repository-name my-app --region us-east-1
```

## Start web server
uvicorn predict:app --host 0.0.0.0 --port 8000 --reload