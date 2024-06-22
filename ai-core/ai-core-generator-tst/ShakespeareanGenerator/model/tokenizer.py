
from tokenizers import SentencePieceBPETokenizer
from ShakespeareanGenerator.parameters import TrainingParameters

class Tokenizer:

    def __init__(self):
        self.training_params = TrainingParameters()
        self.tokenizer = None

    def load_tokenizer(self):
        self.tokenizer = SentencePieceBPETokenizer.from_file(
            self.training_params.INPUT_TOKENIZER_MODEL +'vocab.json', 
            merges_filename = self.training_params.INPUT_TOKENIZER_MODEL +'merges.txt')
        self.tokenizer.add_special_tokens(["<ME>","<STYLE_SHIFT>","<SK>"])

    def encode(self, text):
        if not isinstance(text, str):
            raise TypeError('Input text must be a string.')
        try:
            if self.tokenizer is None:
                self.load_tokenizer()
            return self.tokenizer.encode(text)
        except Exception as e:
            print('Error occurred during encoding: {}'.format(e))
            raise

    def decode(self, text):
        if not isinstance(text, list):
            raise TypeError('Input tokens must be a list.')
        try:
            if self.tokenizer is None:
                self.load_tokenizer()
            return self.tokenizer.decode(text, skip_special_tokens=False)
        except Exception as e:
            print('Error occurred during encoding: {}'.format(e))
            raise