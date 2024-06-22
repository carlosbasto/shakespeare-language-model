# Specify which base layers (default dependencies) to use
# You may find more base layers at https://hub.docker.com/
FROM python:3.9-slim
#
# Creates directory within Docker image
RUN mkdir -p /app/src/

# SAP AI Core will use it as described in the workflow
RUN mkdir -p /app/logs/

# Copies file from Local system TO path indocke Docker image
COPY main.py /app/src/
COPY requirements.txt /app/src/ 
COPY /ShakespeareanGenerator/*.py /app/src/ShakespeareanGenerator/

# Installs dependencies within you Docker image
RUN pip3 install -r /app/src/requirements.txt

# Enable permission to execute anything inside the folder app
RUN chgrp -R 65534 /app && \
    chmod -R 777 /app