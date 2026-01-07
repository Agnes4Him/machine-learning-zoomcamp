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

echo "Waiting for application to become healthy..."
for i in {1..20}; do
  if curl -fs http://localhost:8000/ping; then
    echo "Application is healthy"
    break
  fi
  echo "Not ready yet... retrying ($i)"
  sleep 3
done

# Final check (fails pipeline if still unhealthy)
curl -f http://localhost:8000/ping