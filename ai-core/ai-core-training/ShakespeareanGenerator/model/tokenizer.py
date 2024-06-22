
from tokenizers import SentencePieceBPETokenizer
from ShakespeareanGenerator.parameters import TrainingParameters

class Tokenizer:

    def __init__(self, corpus, vocab_size):
        training_params = TrainingParameters()
        self.TOKENIZER_MODEL_PATH = training_params.TOKENIZER_MODEL_PATH
        self.sentences = corpus.split('\n')
        self.vocab_size = vocab_size
        self.tokenizer = None

    def train_tokenizer(self):
        special_tokens = ["<pad>", "<unk>", "<s>", "</s>", "<b>"]
        self.tokenizer = SentencePieceBPETokenizer()
        self.tokenizer.train_from_iterator(
            self.sentences,
            vocab_size = self.vocab_size,
            min_frequency = 2,
            special_tokens = special_tokens,
            show_progress = False
        )
        self.tokenizer.save_model(self.TOKENIZER_MODEL_PATH)
        
    def encode(self, text):
        if not isinstance(text, str):
            raise TypeError('Input text must be a string.')
        try:
            if self.tokenizer is None:
                self.train_tokenizer()
            return self.tokenizer.encode(text)
        except Exception as e:
            print('Error occurred during encoding: {}'.format(e))
            raise

    def decode(self, text):
        if not isinstance(text, list):
            raise TypeError('Input tokens must be a list.')
        try:
            if self.tokenizer is None:
                self.train_tokenizer()
            return self.tokenizer.decode(text)
        except Exception as e:
            print('Error occurred during encoding: {}'.format(e))
            raise