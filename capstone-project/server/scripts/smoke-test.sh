#!/usr/bin/env bash
set -euo pipefail

DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:-my_dockerhub_username}"
IMAGE_NAME="${IMAGE_NAME:-energy-prediction-server}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

IMAGE="$DOCKERHUB_USERNAME/$IMAGE_NAME:$IMAGE_TAG"

cleanup() {
  echo "Cleaning up containers..."
  docker rm -f smoke-test mlflow-server >/dev/null 2>&1 || true
}

trap cleanup EXIT

echo "Starting MLFLow server..."
docker run -d \
  -p 5000:5000 \
  --name mlflow-server \
  ghcr.io/mlflow/mlflow \
  mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --host 0.0.0.0 \
    --port 5000 \
    --cors-allowed-origins "*" \
    --x-frame-options NONE \
    --disable-security-middleware

echo "Starting container for smoke test..."
docker run -d -p 8000:8000 --name smoke-test "$IMAGE"

sleep 5

echo "Checking health endpoint..."
curl -f http://localhost:8000/ping

echo "Smoke test passed"
