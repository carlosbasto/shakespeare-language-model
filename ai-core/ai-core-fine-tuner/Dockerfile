# Use the PyTorch image with CUDA 12.1 and cuDNN 8 runtime
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories within the Docker image
RUN mkdir -p /app/src /app/data /app/input_model /app/input_tokenizer /app/model /app/logs

# Copy files from local system to path in Docker image
COPY main.py /app/src/
COPY requirements.txt /app/src/
COPY /ShakespeareanGenerator/*.py /app/src/ShakespeareanGenerator/
COPY /ShakespeareanGenerator/model/*.py /app/src/ShakespeareanGenerator/model/

# Install Python dependencies within the Docker image
RUN pip3 install --no-cache-dir -r /app/src/requirements.txt

# Set permissions to execute anything inside the /app folder
RUN chgrp -R 65534 /app && \
    chmod -R 777 /app