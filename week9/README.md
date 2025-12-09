## Library Installations
* keras-image-helper

* onnxruntime

* numpy

## Docker image

1. Build Docker image

```bash
docker build -t model-prediction:1.0.0 .
```


![Build Docker](./images/docker-build.png)


![Architecture](./images/list-images.png)

2. Run Docker container

```bash
docker run -it --rm -p 8080:8080  model-prediction:1.0.0
```

![Run Container](./images/docker-run.png)

## Run prediction locally

```bash
uv run python3 test_predict.py
```

![Test](./images/test-prediction.png)


## ECR

1. Set variables

```bash
AWS_REGION=us-east-1
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REPO_NAME=model-prediction
TAG=1.0.0
```

![Set Env Variables](./images/set-variables.png)

2. Create ECR repository

```bash
aws ecr create-repository \
  --repository-name "$REPO_NAME" \
  --region "$AWS_REGION"
```


![Create ECR](./images/create-ecr.png)

3. Authenticate to ECR

```bash
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login \
      --username AWS \
      --password-stdin "$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
```


![ECR Login](./images/ecr-login.png)

4. Tag docker image for ECR

```bash
docker tag $REPO_NAME:$TAG \
  $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME:$TAG
```


![Docker Tag](./images/docker-tag.png)

5. Push image to ECR

```bash
docker push $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME:$TAG
```

6. Create Lambda function

![Lambda](./images/lambda-console.png)


![Create Lambda](./images/create-lambda.png)


![Lambda2](./images/function-overview.png)


![Function](./images/edit-function.png)


![Test Function](./images/test-function.png)


![Test Output](./images/test-function-output.png)


7. Create API Gateway


![API Gateway](./images/api-gw-console.png)


![API GW](./images/api-gw-type.png)


![API GW3](./images/create-api-gw.png)


![API GW4](./images/create-gw-resource.png)


![API GW5](./images/gw-resource-overview.png)


![API GW6](./images/gw-method.png)


![API GW7](./images/gw-method-overview.png)


![API GW8](./images/gw-stage-overview.png)


8. Test predictions


![Test Prediction](./images/gw-testing.png)


![Test Prediction - Output](./images/gw-testing-output.png)
