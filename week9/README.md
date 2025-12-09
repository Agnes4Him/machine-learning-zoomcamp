## Installations
pip install keras-image-helper

pip install onnxruntime

pip install numpy

## ECR
aws ecr get-login-password --region us-east-1 | \
docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

docker tag teen-addiction-prediction:1.0.0 <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/<IMAGE_NAME>:<TAG>

docker push <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/<IMAGE_NAME>:<TAG>