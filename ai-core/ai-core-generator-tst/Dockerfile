
# Use the PyTorch image with CUDA 12.1 and cuDNN 8 runtime
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Set the locale to avoid issues with encoding
ENV LANG C.UTF-8

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create the application directory within the Docker image
RUN mkdir -p /app/src

# Copy application files from local system to the Docker image
COPY main.py /app/src/
COPY requirements.txt /app/src/
COPY ShakespeareanGenerator/*.py /app/src/ShakespeareanGenerator/
COPY ShakespeareanGenerator/model/*.py /app/src/ShakespeareanGenerator/model/

# Install Python dependencies within the Docker image
RUN pip3 install --no-cache-dir -r /app/src/requirements.txt

# Set appropriate permissions for the application directory
RUN chgrp -R nogroup /app && \
    chmod -R 770 /app