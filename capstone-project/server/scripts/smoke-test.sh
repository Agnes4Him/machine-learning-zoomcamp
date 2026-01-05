#!/usr/bin/env bash
set -e

IMAGE="$DOCKERHUB_USERNAME/fastapi-web:$GITHUB_SHA"

echo "Starting container for smoke test..."
docker run -d -p 8000:8000 --name smoke-test "$IMAGE"

sleep 5

echo "Checking health endpoint..."
curl -f http://localhost:8000/ping

echo "Smoke test passed"

docker stop smoke-test
docker rm smoke-test