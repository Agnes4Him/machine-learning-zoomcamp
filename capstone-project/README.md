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

### Create K8s cluster with `kind`
```bash
kind create cluster --name ml --config kubernetes/kind/kind-config.yaml
```
