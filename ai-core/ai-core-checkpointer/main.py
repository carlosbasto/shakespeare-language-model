
import pickle
import torch
from ShakespeareanGenerator.model.language_models import ShakespeareanLanguagelModel, ModelTrainer
from ShakespeareanGenerator.parameters import TrainingParameters
from ShakespeareanGenerator.data_handler import DataHandler
from ShakespeareanGenerator.logger import Logger

class Run:
    def __init__(self):
        self.logging = Logger()
        self.training_params = TrainingParameters()
        self.check_gpu_usage()
        self.prepare_data()
        self.train_model()

    def check_gpu_usage(self):
        if torch.cuda.is_available():
            self.logging.info(f"GPU is available, using GPU: {torch.cuda.get_device_name(0)}")
            self.logging.info(f"Using CUDA version {torch.version.cuda}")
        else:
            self.logging.warning("GPU is not available, using CPU.")        

    def prepare_data(self):
        self.logging.info('START OF EXECUTION')
        self.logging.info('Get DataHandler and Model Instances')
        self.data_handler = DataHandler(self.training_params.DATA_PATH)
        try:
            with open(self.training_params.INPUT_MODEL + 'model.pkl', 'rb') as f:
                loaded_model = pickle.load(f)
                self.logging.info('Loaded model for continuing training')
        except FileNotFoundError:
            loaded_model = None
            self.logging.error('Transfer learning not possible; no model found')
            self.logging.warning('Model will start from scratch')
        
        self.model_object = ShakespeareanLanguagelModel()
        model = self.model_object if loaded_model is None else loaded_model
        self.model = model.to(self.training_params.device)
        self.logging.info('DataHandler and Model Instantiated')        

    def train_model(self):
        self.trainer = ModelTrainer(self.data_handler, self.model)
        self.trainer.train()
        self.logging.info('Model was trained successfully')        
        with open(self.training_params.MODEL_PATH + 'model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        self.logging.info('END OF EXECUTION')

if __name__ == '__main__':
    Run()