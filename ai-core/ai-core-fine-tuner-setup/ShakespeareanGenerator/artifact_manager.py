import boto3
import requests
from ShakespeareanGenerator.logger import Logger
from ShakespeareanGenerator.parameters import ObjectStoreParameters

class ObjectStoreArtifactManager:

    def __init__(self):

        self.logging = Logger()
        self.obj_parameters = ObjectStoreParameters()
        self.s3 = self.__get_s3_connection()
        self.latest_execution_id = None

    def __get_s3_connection(self):
        return boto3.client(
            's3',
            aws_access_key_id = self.obj_parameters.access_key_id,
            aws_secret_access_key = self.obj_parameters.secret_access_key
            )    
    
    def __get_executions(self):
        response = self.s3.list_objects_v2(Bucket=self.obj_parameters.bucket_name, Prefix=self.obj_parameters.prefix_m)
        unique_prefixes = set()
        for obj in response['Contents']:
            prefix_part = obj['Key'].split('/')[2]
            unique_prefixes.add(prefix_part)
        sorted_objects = sorted(response['Contents'], key=lambda x: x['LastModified'])
        latest_keys = {}
        for obj in sorted_objects:
            prefix_part = obj['Key'].split('/')[2]
            if prefix_part not in latest_keys:
                latest_keys[prefix_part] = obj['Key'].split('/')[2]
        
        self.sorted_keys = list(latest_keys.values())
               
    def __check_model_files_exist(self):
        source_key = f"{self.obj_parameters.prefix_m}{self.latest_execution_id}/{self.model_type}/"
        response = self.s3.list_objects_v2(Bucket=self.obj_parameters.bucket_name, Prefix=source_key)

        if 'Contents' not in response:
            self.logging.warning(f'No {self.model_type} files found in {source_key}')
            return False

        # Check for existence of 'model.pkl' file if model_type is 'model'
        if self.model_type == 'model':
            desired_key = f"{source_key}model.pkl"
            model_file_exists = any(obj['Key'] == desired_key for obj in response['Contents'])
            if not model_file_exists:
                self.logging.warning(f'{desired_key} not found in {source_key}')
                return False

        return True
    
    def __get_latest_valid_execution_id(self):

        if not hasattr(self, 'sorted_keys'):
            self.__get_executions()
            self.logging.info('Reading all the models in object store from all executions')
        if not hasattr(self, 'current_index'):
            self.current_index = 0 
            self.logging.info(f'Initial Index: {self.current_index}')
        reversed_prefixes = list(map(lambda x: x, reversed(self.sorted_keys)))
        for index in range(0, len(self.sorted_keys)):
            self.latest_execution_id = reversed_prefixes[index]
            if self.__check_model_files_exist():
                return self.latest_execution_id
            else:
                msg = 'Files for execution ID not found. {}'.format(self.latest_execution_id)
                self.logging.warning(msg)    

    def copy_object(self, model_type):
        
        self.model_type = model_type
        self.__get_latest_valid_execution_id()
        model_mappings = {
            'model': ('model.pkl','{}{}model.pkl'.format(
                self.obj_parameters.prefix,
                self.obj_parameters.INPUT_MODEL_PATH
            )),
            'bpe_model_vocab': ('vocab.json', '{}{}vocab.json'.format(
                self.obj_parameters.prefix,
                self.obj_parameters.INPUT_BPE_MODEL_PATH
            )),
            'bpe_model_merges': ('merges.txt', '{}{}merges.txt'.format(
                self.obj_parameters.prefix,
                self.obj_parameters.INPUT_BPE_MODEL_PATH
            ))
        }
        if not any(key.startswith(model_type) for key in model_mappings):
            raise ValueError(f"Invalid model_type: {model_type}")
        for key, (model_file_name, destination_key) in model_mappings.items():
            if key.startswith(model_type):
                source_key = f"{self.obj_parameters.prefix_m}{self.latest_execution_id}/{model_type}/{model_file_name}"  
                self.logging.info(f'FROM: {source_key} TO: {destination_key}')
                self.logging.info(f'Starting copy process for {model_type}')
                self.s3.copy_object(
                    Bucket=self.obj_parameters.bucket_name,
                    CopySource={'Bucket': self.obj_parameters.bucket_name, 'Key': source_key},
                    Key=destination_key
                )
                self.logging.info(f'{model_type} artifacts were updated from {self.latest_execution_id} folder to the input folders for further processing')
        return self.latest_execution_id

    def upload_file_to_object_store(self):
        url = "https://raw.githubusercontent.com/carlosbasto/shakespeare-language-model/main/ai-core/ai-core-datasets/shakespeare-style-transfer.txt"
        
        file_key = f"{self.obj_parameters.prefix}{self.obj_parameters.DATA_PATH + self.obj_parameters.DATA_NAME}"
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            corpus = response.text
            self.s3.put_object(
                Bucket=self.obj_parameters.bucket_name,
                Key=file_key,
                Body=corpus.encode('utf-8')
            )
            self.logging.info(f"Uploaded shakespeare-style-transfer.txt to S3 path: {file_key}")
            self.logging.info(f"{self.obj_parameters.prefix_m}")
        except requests.RequestException as e:
            error_msg = f"Error fetching data from URL: {e}"
            print(error_msg)
            self.logging.error(error_msg)
        except Exception as e:
            error_msg = f"An unexpected error occurred: {e}"
            print(error_msg)
            self.logging.error(error_msg)
