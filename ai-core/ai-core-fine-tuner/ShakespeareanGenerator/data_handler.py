import torch
from ShakespeareanGenerator.model.tokenizer import Tokenizer
from ShakespeareanGenerator.parameters import TrainingParameters
from ShakespeareanGenerator.logger import Logger

class DataHandler:

    def __init__(self, path):
        self.logging = Logger()
        self.training_params = TrainingParameters()
        self.path = path
        self.data = None

    def get_data(self):
        try:
            with open(self.path, 'r', encoding='utf-8') as file:
                data = file.read()
            lines = data.splitlines()
            formatted_sentences = []
            for line in lines:
                modern, shakespearean = line.split(';')
                formatted_sentence = f'<ME>{modern.strip()}<STYLE_SHIFT>{shakespearean.strip()}<SK>'
                formatted_sentences.append(formatted_sentence)
            self.data = formatted_sentences
        except FileNotFoundError:
            msg = 'File {} not found.'.format(self.path)
            self.logging.error(msg)
            raise FileNotFoundError(msg)
        

    def get_batch(self, split):
        
        if self.data is None:
            self.get_data()
        tokenizer = Tokenizer()
        
        # Encode each sentence individually and concatenate them
        encoded_corpus = []
        for sentence in self.data:
            encoded_sentence = tokenizer.encode(sentence)
            encoded_corpus.extend(encoded_sentence.ids)
        
        data = torch.tensor(encoded_corpus, dtype=torch.long)

        split_point = int(0.9 * len(data))
        training_set, validation_set = data[:split_point], data[split_point:]
        selected_data = training_set if split == 'train' else validation_set
        indices = torch.randint(len(selected_data) - self.training_params.context_length, (self.training_params.batch_size,))

        batches_x = []
        batches_y = []
        for index in indices:
            batch_x = selected_data[index:index + self.training_params.context_length]
            batch_y = selected_data[index + 1:index + self.training_params.context_length + 1]
            batches_x.append(batch_x)
            batches_y.append(batch_y)

        x = torch.stack(batches_x)
        y = torch.stack(batches_y)
        x, y = x.to(self.training_params.device), y.to(self.training_params.device)
        return x, y
    

    @torch.no_grad()
    def get_estimated_loss(self, model):
        out = {}
        model.eval()
        
        for split in ['train', 'val']:
            losses = torch.zeros(self.training_params.eval_steps)
            for k in range(self.training_params.eval_steps):
                X, Y = self.get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
            self.logging.info('Estimated losses: {}'.format(losses.mean()))
        model.train()
        return out
