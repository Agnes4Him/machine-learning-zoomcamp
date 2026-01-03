FROM python:3.12.3-slim-bookworm

RUN pip install fastapi uvicorn mlflow dotenv

COPY ["predict.py", "./"]

EXPOSE 8000

ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]