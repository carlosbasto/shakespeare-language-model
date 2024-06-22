import logging
import boto3
import threading
import tempfile
from ShakespeareanGenerator.parameters import LogParameters

class Logger:
    def __init__(self):
        self.log_params = LogParameters()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.temp_file = tempfile.NamedTemporaryFile(mode='a', delete=False)
        self.file_handler = logging.FileHandler(self.temp_file.name)
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s | %(name)s → %(levelname)s: %(message)s'))
        self.logger.addHandler(self.file_handler)
        self.s3 = self.__get_s3_connection()
        self.upload_logs_to_s3()

    def __get_s3_connection(self):
        return boto3.client(
            's3',
            aws_access_key_id=self.log_params.access_key_id,
            aws_secret_access_key=self.log_params.secret_access_key
        ) 

    def upload_logs_to_s3(self):
        try:
            # Read logs from the temporary file
            with open(self.temp_file.name, 'r') as f:
                log_body = f.read().strip()

            if log_body:
                file_key = self.log_params.log_prefix + self.log_params.LOG_NAME
                self.s3.put_object(
                    Bucket=self.log_params.bucket_name,
                    Key=file_key,
                    Body=log_body.encode('utf-8')
                )
            else:
                self.logger.info("No logs to upload.")
        except Exception as e:
            self.logger.error(f"Error uploading log to S3: {e}")

        # Reschedule the timer for the next upload
        self.schedule_next_upload()

    def schedule_next_upload(self):
        # Create a new timer for the next upload after the specified interval
        self.upload_timer = threading.Timer(self.log_params.upload_interval, self.upload_logs_to_s3)
        self.upload_timer.start()

    def log(self, level, message):
        getattr(self.logger, level)(message)

    def info(self, message):
        self.log('info', message)

    def warning(self, message):
        self.log('warning', message)

    def error(self, message):
        self.log('error', message)

    def critical(self, message):
        self.log('critical', message)
