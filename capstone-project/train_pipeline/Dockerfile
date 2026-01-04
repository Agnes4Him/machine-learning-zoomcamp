# Use an official Python runtime as the base image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install the required packages
RUN pip install prefect mlflow scikit-learn pandas numpy python-dotenv

# Copy the rest of the application code
COPY pipeline.py .

# Set the command to run your application
CMD ["python", "pipeline.py"]