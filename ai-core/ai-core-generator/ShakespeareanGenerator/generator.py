import io
import torch
import pickle
from torch.nn import functional as F
from ShakespeareanGenerator.model.tokenizer import Tokenizer
from ShakespeareanGenerator.parameters import ServingParameters
from ShakespeareanGenerator.logger import Logger

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

class ModelManager:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.serving_params = ServingParameters()
        self.logging = Logger()
        self.check_gpu_usage()

    def check_gpu_usage(self):
        if torch.cuda.is_available():
            self.logging.info(f"GPU is available, using GPU: {torch.cuda.get_device_name(0)}")
            self.logging.info(f"Using CUDA version {torch.version.cuda}")
        else:
            self.logging.warning("GPU is not available, using CPU.")

    def load_model(self):
        if self.serving_params.device == 'cpu':
            with open(self.serving_params.INPUT_MODEL, 'rb') as f:
                self.model = CPU_Unpickler(f).load()  # no-gpu based testing
        elif self.serving_params.device == 'cuda':
            with open(self.serving_params.INPUT_MODEL, 'rb') as f:
                self.model = pickle.load(f)
                
        self.model.eval()
        self.model = self.model.to(self.serving_params.device)
        self.logging.info(f"Model loaded and sent to {self.serving_params.device}")
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

    def __sample_from_model(self, index):
        self.model = self.model_manager.model
        for _ in range(self.max_tokens):
            try:
                current_index = index[:, -self.model.position_embeddings.weight.shape[0]:]
                logits, _ = self.model(current_index)
                scaled_logits = (lambda l, t: l / t if t > 0.0 else l)(logits[:, -1, :], self.temperature)
                probs = F.softmax(scaled_logits, dim=-1)

                if self.top_k > 0:
                    probs_value, probs_indices = torch.topk(probs, self.top_k, dim=-1)
                    filtered_probs = probs.clone().fill_(0.0)
                    filtered_probs.scatter_(dim=-1, index=probs_indices, src=probs_value)
                    probs = filtered_probs / torch.sum(filtered_probs, dim=-1, keepdim=True)

                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                sorted_indices_to_remove = cumulative_probs > self.top_p
                if torch.any(sorted_indices_to_remove):
                    cutoff_idx = torch.where(sorted_indices_to_remove)[1][0]
                    indices_to_remove = sorted_indices[:, cutoff_idx + 1 :]
                    probs.scatter_(dim=-1, index=indices_to_remove, value=0.0)
                probs = probs / torch.sum(probs, dim=-1, keepdim=True)

                next_index = torch.multinomial(probs, num_samples=1)
                index = torch.cat((index, next_index), dim=1)
            except Exception as e:
                self.model_manager.logging.error(f"Error during text generation: {str(e)}")
                raise
        return index

    def post_process_text(self, generated_text):
        cleaned_text = generated_text.replace("<s>", "").replace("</s>", "").replace("<b>", "").strip()
        return cleaned_text

    with torch.inference_mode():
        def generate(self):
            if not self.model_manager.is_model_loaded():
                self.model_manager.load_model()
            try:
                idx = torch.full((1, 1), 4, dtype=torch.long, device=self.model_manager.serving_params.device)
                completion = self.tokenizer.decode(self.__sample_from_model(idx)[0].tolist())
                self.length = len(self.tokenizer.encode(completion).ids)
                self.model_manager.logging.info(f"Text generated successfully with length: {self.length}")
                self.model_manager.logging.info(f"With max tokens set to: {self.max_tokens}")
                self.model_manager.logging.info(f"With temperature set to: {self.temperature}")
                self.model_manager.logging.info(f"With top k set to: {self.top_k}")
                self.model_manager.logging.info(f"With top p set to: {self.top_p}")
                return completion
            except Exception as e:
                self.model_manager.logging.error(f"Error during text generation: {str(e)}")
                raise
