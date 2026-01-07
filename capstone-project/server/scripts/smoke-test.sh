#!/usr/bin/env bash
set -e

DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:-my_dockerhub_username}"
IMAGE_NAME="${IMAGE_NAME:-energy-prediction-server}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

IMAGE="$DOCKERHUB_USERNAME/$IMAGE_NAME:$IMAGE_TAG"

echo "Starting MLFLow server..."
mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000 \
--cors-allowed-origins "*" \
--x-frame-options NONE \
--disable-security-middleware

echo "Starting container for smoke test..."
docker run -d -p 8000:8000 --name smoke-test "$IMAGE"

sleep 5

echo "Checking health endpoint..."
curl -f http://localhost:8000/ping

echo "Smoke test passed"

docker stop smoke-test
docker rm smoke-test