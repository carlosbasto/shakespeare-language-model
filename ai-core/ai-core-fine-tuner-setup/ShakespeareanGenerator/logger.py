
import logging
from ShakespeareanGenerator.parameters import ObjectStoreParameters

class Logger:
    def __init__(self):
        
        self.obj_params = ObjectStoreParameters()
        self.log_file = self.obj_params.LOG_PATH + self.obj_params.LOG_NAME
        
        logging.basicConfig(
            filename=self.log_file,
            filemode='w',
            format='%(asctime)s | %(name)s → %(levelname)s: %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)
        
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
