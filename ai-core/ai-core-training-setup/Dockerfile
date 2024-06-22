# Use a slim Python 3.9 image as the base layer to minimize the image size
FROM python:3.9-slim

# Create necessary directories within the Docker image
# /app/src: Directory for source code
# /app/logs: Directory for log files
RUN mkdir -p /app/src /app/logs

# Copy the main Python script and requirements file from the local system to the Docker image
COPY main.py /app/src/
COPY requirements.txt /app/src/

# Copy the ShakespeareanGenerator module from the local system to the Docker image
COPY /ShakespeareanGenerator/*.py /app/src/ShakespeareanGenerator/

# Install the required Python packages specified in requirements.txt
# --no-cache-dir: Do not cache the packages, reducing the image size
RUN pip3 install --no-cache-dir -r /app/src/requirements.txt

# Change the group ownership of all files and directories under /app to the group ID 65534 (typically 'nogroup')
# Set read, write, and execute permissions for all users on all files and directories under /app
RUN chgrp -R 65534 /app && \
    chmod -R 777 /app