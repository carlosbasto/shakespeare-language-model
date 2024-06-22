import boto3
import requests
from ShakespeareanGenerator.logger import Logger
from ShakespeareanGenerator.parameters import ObjectStoreParameters

class ObjectStoreArtifactManager:

    def __init__(self):
        self.logging = Logger()
        self.obj_parameters = ObjectStoreParameters()
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=self.obj_parameters.access_key_id,
            aws_secret_access_key=self.obj_parameters.secret_access_key
        )

    def upload_file_to_object_store(self):
        url = "https://raw.githubusercontent.com/carlosbasto/shakespeare-language-model/main/ai-core/ai-core-datasets/tinyshakespeare.txt"
        
        file_key = f"{self.obj_parameters.prefix}{self.obj_parameters.DATA_PATH + self.obj_parameters.DATA_NAME}"
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            corpus = response.text
            corpus = "<b>".join(corpus.split('\n'))
            self.s3.put_object(
                Bucket=self.obj_parameters.bucket_name,
                Key=file_key,
                Body=corpus.encode('utf-8')
            )
            self.logging.info(f"Uploaded tinyshakespeare.txt to S3 path: {file_key}")
        except requests.RequestException as e:
            error_msg = f"Error fetching data from URL: {e}"
            print(error_msg)
            self.logging.error(error_msg)
        except Exception as e:
            error_msg = f"An unexpected error occurred: {e}"
            print(error_msg)
            self.logging.error(error_msg)
