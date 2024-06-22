# Import necessary libraries
import os
import torch

class TrainingParameters:
    def __init__(self):
        # PARAMETERS AVAILABLE FOR UNPICKLE PRE-TRAINED MODEL
        # context_length is tied to the architecture of the pre-trained model
        #self.context_length = int(os.environ.get('CONTEXT_LENGTH'))
        self.context_length = 256
        # The embedding dimension and attention heads are core architectural 
        # parameters. Changing it would require reinitializing the embedding 
        # layer and other dependent components, which would negate the benefits 
        # of pre-training.
        #self.embedding_dim = int(os.environ.get('EMBEDDING_DIM'))
        self.embedding_dim = 384
        #self.attention_heads = int(os.environ.get('ATTENTION_HEADS'))
        self.attention_heads = 6
        # The number of layers in the model defines its depth. Changing this 
        # would require adding or removing layers, which is not compatible 
        # with the pre-trained model architecture 
        #self.num_layers = int(os.environ.get('NUM_LAYERS'))
        self.num_layers = 6
        # The size of the dictionary (vocabulary size) must remain the same
        # to ensure that the embeddings and subsequent layers correctly map 
        # the input tokens. Changing this would require reinitializing the 
        # embedding layer.
        #self.dictionary_size = int(os.environ.get('DICTIONARY_SIZE'))
        self.dictionary_size = 65 + 3 + 6 # 3 new special chars from fine tuning 
                                          # and 6 from training
                
        #________________________________           
        # SELECTABLE PARAMETERS FOR FINE TUNING
        # it can adjust the batch size during fine-tuning based on your 
        # computational resources and the stability of training
        self.batch_size = int(os.environ.get('BATCH_SIZE'))
        self.iteration_limit = int(os.environ.get('ITERATION_LIMIT'))
        self.eval_frequency = int(os.environ.get('EVAL_FREQUENCY'))
        self.eval_steps = int(os.environ.get('EVAL_STEPS'))
        # The learning rate for fine-tuning should generally be lower than 
        # the learning rate used during pre-training
        self.learning_rate = float(os.environ.get('LEARNING_RATE'))
        self.dropout = float(os.environ.get('DROPOUT'))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.DATA_PATH = '/app/data/shakespeare-style-transfer.txt'
        self.MODEL_PATH = '/app/model/'
        self.TOKENIZER_MODEL_PATH = '/app/tokenizer/'
        self.LOG_PATH = '/app/logs/'
        self.LOG_NAME = 'fine_tuning_logs.log'
        self.INPUT_MODEL = '/app/input_model/'
        self.INPUT_TOKENIZER_MODEL = '/app/input_tokenizer/'

class LoraParameters:
    def __init__(self):
        self.r = int(os.environ.get('LORA_RANK'))
        self.alpha = int(os.environ.get('LORA_ALPHA'))