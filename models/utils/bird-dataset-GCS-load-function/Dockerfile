# Use a more specific base image with platform specification
FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Add executable permissions
RUN chmod +x /app/main.py

# Test Python execution
RUN python -c "import sys; print(sys.version)"

# Set the entry point
ENTRYPOINT ["python", "main.py"]
