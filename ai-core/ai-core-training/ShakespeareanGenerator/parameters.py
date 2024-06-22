# Import necessary libraries
import os
import torch

class TrainingParameters:
    def __init__(self):
        self.batch_size = int(os.environ.get('BATCH_SIZE'))
        self.context_length = int(os.environ.get('CONTEXT_LENGTH'))
        self.iteration_limit = int(os.environ.get('ITERATION_LIMIT'))
        self.eval_frequency = int(os.environ.get('EVAL_FREQUENCY'))
        self.eval_steps = int(os.environ.get('EVAL_STEPS'))
        self.learning_rate = float(os.environ.get('LEARNING_RATE'))
        self.embedding_dim = int(os.environ.get('EMBEDDING_DIM'))
        self.attention_heads = int(os.environ.get('ATTENTION_HEADS'))
        self.num_layers = int(os.environ.get('NUM_LAYERS'))
        self.dropout = float(os.environ.get('DROPOUT'))
        self.dictionary_size = int(os.environ.get('DICTIONARY_SIZE')) + 6
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.DATA_PATH = '/app/data/tinyshakespeare.txt'
        self.MODEL_PATH = '/app/model/'
        self.TOKENIZER_MODEL_PATH = '/app/tokenizer/'
        self.LOG_PATH = '/app/logs/'
        self.LOG_NAME = 'train_logs.log'
