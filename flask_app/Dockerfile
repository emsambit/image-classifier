# Use Python 3.9 as the base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY ./flask_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask app code
COPY ./flask_app/ .

# Copy the joblib directory from the root build context
COPY ./joblib ./joblib

# Expose port 5000
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
