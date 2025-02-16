# Use an official lightweight Python image.
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Prevent apt-get from prompting for input.
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory.
WORKDIR /app

# Install system dependencies.
RUN apt-get update -o Acquire::Retries=3 && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency definitions.
COPY src/requirements.txt src/setup.py ./

# Upgrade pip and install Python dependencies,
# then install your package in editable mode.
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -e .

# Copy the rest of the application code.
COPY src/ .

# Define the default command to run your app.
CMD ["python", "main.py"]
