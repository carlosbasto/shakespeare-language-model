# Import necessary libraries
import os

class ObjectStoreParameters:
    def __init__(self):
        self.bucket_name = os.environ.get('BUCKET_NAME')
        self.prefix = os.environ.get('PREFIX_NAME') # 'shakespeare/repository/'
        self.prefix_m = self.prefix.split('/')[0] + '/executions/'
        self.access_key_id = os.environ.get('ACCESS_KEY_ID')
        self.secret_access_key = os.environ.get('SECRET_ACCESS_KEY')
        self.DATA_PATH = 'data/'
        self.DATA_NAME = 'tinyshakespeare.txt'        
        self.LOG_PATH = '/app/logs/'
        self.LOG_NAME = 'setup_logs.log'

        self.INPUT_MODEL_PATH = 'input_model/'
        self.INPUT_BPE_MODEL_PATH = 'input_tokenizer/'