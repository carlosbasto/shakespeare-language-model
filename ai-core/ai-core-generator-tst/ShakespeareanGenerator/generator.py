import io
import re
import torch
import pickle
from torch.nn import functional as F
from ShakespeareanGenerator.model.tokenizer import Tokenizer
from ShakespeareanGenerator.parameters import TrainingParameters
from ShakespeareanGenerator.logger import Logger

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

class ModelManager:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.training_params = TrainingParameters()
        self.logging = Logger()
        self.check_gpu_usage()

    def check_gpu_usage(self):
        if torch.cuda.is_available():
            self.logging.info(f"GPU is available, using GPU: {torch.cuda.get_device_name(0)}")
            self.logging.info(f"Using CUDA version {torch.version.cuda}")
        else:
            self.logging.warning("GPU is not available, using CPU.")

    def load_model(self):
        if self.training_params.device == 'cpu':
            with open(self.training_params.INPUT_MODEL, 'rb') as f:
                self.model = CPU_Unpickler(f).load()  # no-gpu based testing
        elif self.training_params.device == 'cuda':
            with open(self.training_params.INPUT_MODEL, 'rb') as f:
                self.model = pickle.load(f)
                
        self.model.eval()
        self.model = self.model.to(self.training_params.device)
        self.logging.info(f"Model loaded and sent to {self.training_params.device}")
        self.model_loaded = True

    def is_model_loaded(self):
        return self.model_loaded

class Generator:
    def __init__(self, model_manager, max_tokens, temperature, top_k, top_p):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.tokenizer = Tokenizer()
        self.model_manager = model_manager


    def __apply_sample_methods(self, probs):
        if self.top_k > 0:
            probs_value, probs_indices = torch.topk(probs, self.top_k, dim=-1)
            filtered_probs = probs.clone().fill_(0.0)
            filtered_probs.scatter_(dim=-1, index=probs_indices, src=probs_value)
            filtered_probs = filtered_probs / torch.sum(filtered_probs, dim=-1, keepdim=True)
            next_token = torch.multinomial(filtered_probs, 1)

        elif self.top_p > 0.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=1)
            exceed_threshold = cumulative_probs > self.top_p
            exceed_threshold[:, 1:] = exceed_threshold[:, :-1].clone()
            exceed_threshold[:, 0] = False
            sorted_probs[exceed_threshold] = 0
            sampled_index = torch.multinomial(sorted_probs, 1)
            next_token = sorted_indices.gather(dim=1, index=sampled_index)
            

        else:
            next_token = torch.multinomial(probs, 1)
        return next_token


    def __sample_from_model(self, index):
        self.model = self.model_manager.model
        context_length = 256
        for _ in range(self.max_tokens):
            try:
                current_index = index[:, -context_length:]
                logits, _ = self.model(current_index)
                
                logits = logits[:, -1, :] # last token
                
                if self.temperature > 0.0:
                    logits = logits / self.temperature

                probs = F.softmax(logits, dim=-1)

                next_token = self.__apply_sample_methods(probs)
                index = torch.cat((index, next_token), dim=1)
                
                if next_token.item() == self.tokenizer.encode("<SK>").ids[0]:
                    break

            except Exception as e:
                self.model_manager.logging.error(f"Error during text generation: {str(e)}")
                raise
        return index


    def __get_completion(self, sampled_ids, context):
        sequence = sampled_ids[0, context.size(1):].tolist()
        return self.tokenizer.decode(sequence)

    def post_process_text(self, generated_text):
        pattern = r"<ME>|<STYLE_SHIFT>|<SK>|<pad>"
        return re.sub(pattern, "", generated_text).strip()
    

    with torch.inference_mode():
        def generate(self, modern_sentence):

            try:
                input_sequence = "<ME>" + modern_sentence + "<STYLE_SHIFT>"
                enc_context = self.tokenizer.encode(input_sequence)
                context = torch.tensor(enc_context.ids, dtype=torch.long, device=self.model_manager.training_params.device).unsqueeze(0)
                sampled_indices = self.__sample_from_model(context)
                completion = self.__get_completion(sampled_indices, context)

                self.length = len(self.tokenizer.encode(completion).ids)
                print(completion)
                self.model_manager.logging.info(f"Text generated successfully with length: {self.length}")
                self.model_manager.logging.info(f"With max tokens set to: {self.max_tokens}")
                self.model_manager.logging.info(f"With temperature set to: {self.temperature}")
                self.model_manager.logging.info(f"With top k set to: {self.top_k}")
                self.model_manager.logging.info(f"With top p set to: {self.top_p}")

                return self.post_process_text(completion)
            
            except Exception as e:
                self.model_manager.logging.error(f"Error during text generation: {str(e)}")
                raise