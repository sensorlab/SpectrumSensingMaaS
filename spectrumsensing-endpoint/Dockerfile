# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container to /app
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install -r requirements.txt

ADD . /app

EXPOSE 8000

# Run app.py when the container launches
CMD ["uvicorn", "api-endpoint:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--limit-concurrency", "8000", "--log-level", "error", "--backlog", "8000"]