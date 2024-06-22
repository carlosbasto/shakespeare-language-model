import io
import torch
import pickle
import torch.nn as nn
from ShakespeareanGenerator.model.language_models import ShakespeareanLanguagelModel
from ShakespeareanGenerator.model.peft_models import ModelTrainer, PEFTModel
from ShakespeareanGenerator.parameters import TrainingParameters
from ShakespeareanGenerator.data_handler import DataHandler
from ShakespeareanGenerator.logger import Logger

training_params = TrainingParameters()

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

class Run:
    def __init__(self):
        self.logging = Logger()
        self.training_params = TrainingParameters()
        self.check_gpu_usage()
        self.prepare_data()
        self.train_model()

    def check_gpu_usage(self):
        if torch.cuda.is_available():
            self.logging.warning(f"GPU is available. Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.logging.info("GPU is not available. Using CPU.")   

    def is_model_on_gpu(self, model: nn.Module) -> bool:
        return next(model.parameters()).is_cuda                 

    def prepare_data(self):
        self.logging.info('START OF EXECUTION')
        self.logging.info('Get DataHandler and Model Instances')
        self.data_handler = DataHandler(self.training_params.DATA_PATH)
        torch.set_float32_matmul_precision('high') 
        try:
            with open(self.training_params.INPUT_MODEL + 'model.pkl', 'rb') as f:
                pretrained_model = CPU_Unpickler(f).load()
                self.logging.info('Loaded model for fine tuning')
        except FileNotFoundError:
            pretrained_model = None
            self.logging.error('Transfer learning not possible; no model found')
            self.logging.warning('Model will start from scratch')
        
        new_model = ShakespeareanLanguagelModel()
        self.model = new_model if pretrained_model is None else pretrained_model
        self.model.resize_token_embeddings(training_params.dictionary_size)
        self.model = self.model.to(self.training_params.device)
        self.peft_model = PEFTModel(self.model).to(self.training_params.device)
        is_gpu_used = "PEFT model is on GPU." if self.is_model_on_gpu(self.peft_model) else "PEFT model is on CPU."
        self.logging.info(is_gpu_used)
        self.logging.info(f'Fine Tuned Model: \n {self.peft_model}')
        self.logging.info('DataHandler and Model Instantiated')         

    def train_model(self):
        self.trainer = ModelTrainer(self.data_handler, self.peft_model)
        self.trainer.train()
        self.logging.info('Model was trained successfully')
        with open(self.training_params.MODEL_PATH + 'style-transfer-model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        self.logging.info('END OF EXECUTION')

if __name__ == '__main__':
    Run()
