# Import necessary libraries
import os
import torch

class ServingParameters:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.INPUT_MODEL = '/mnt/models/model.pkl'
        self.INPUT_TOKENIZER_MODEL = '/mnt/models/'


class LogParameters:
    def __init__(self):
        self.bucket_name = os.environ.get('BUCKET_NAME')
        self.prefix = os.environ.get('PREFIX_NAME') # 'shakespeare/repository/'
        self.log_prefix = self.prefix + 'deployments/logs/'
        self.access_key_id = os.environ.get('ACCESS_KEY_ID')
        self.secret_access_key = os.environ.get('SECRET_ACCESS_KEY')
        self.upload_interval = 20  # Default upload interval of 20 seconds
        self.LOG_NAME = 'text_generator_logs.log'
        