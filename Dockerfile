FROM python:3.10-slim

WORKDIR /

# Install system dependencies (git for installing index-tts from GitHub)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your handler file
COPY handler.py /

# Start the container
CMD ["python3", "-u", "handler.py"]

