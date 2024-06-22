import torch
import torch.nn as nn
from torch.nn import functional as F
from ShakespeareanGenerator.logger import Logger
from ShakespeareanGenerator.parameters import TrainingParameters, LoraParameters
from ai_core_sdk.tracking import Tracking
from ai_core_sdk.models import Metric, MetricCustomInfo
from datetime import datetime, timezone

# Start tracking
tracking = Tracking()
logging = Logger()
training_params = TrainingParameters()
lora_params = LoraParameters()

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(lora_params.r).float())
        self.A = nn.Parameter(torch.randn(in_dim, lora_params.r) * std_dev)
        self.B = nn.Parameter(torch.zeros(lora_params.r, out_dim))
        self.scaling = lora_params.alpha / lora_params.r

    def forward(self, x):
        return self.scaling * (x @ self.A @ self.B)


class LinearWithLoRA(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features)

    def forward(self, x):
        return self.linear(x) + self.lora(x)

    
class MultiHeadAttentionWithLoRA(nn.Module):
    def __init__(self, original_mha):
        super().__init__()
        self.heads = nn.ModuleList([HeadWithLoRA(original_mha.heads[i]) for i in range(len(original_mha.heads))])
        self.projection = LinearWithLoRA(original_mha.projection)

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        out = torch.cat(head_outputs, dim=-1)
        out = self.projection(out)
        return out


class HeadWithLoRA(nn.Module):
    def __init__(self, original_head):
        super().__init__()
        self.key = LinearWithLoRA(original_head.key)
        self.query = LinearWithLoRA(original_head.query)
        self.value = LinearWithLoRA(original_head.value)
        self.tril = original_head.tril
        self.dropout = nn.Dropout(training_params.dropout)

    def __compute_weights(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        weights = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        return weights

    def forward(self, x):
        weights = self.__compute_weights(x)
        v = self.value(x)
        return weights @ v


class FeedForwardWithLoRA(nn.Module):
    def __init__(self, original_ffn):
        super().__init__()
        self.ffnet = nn.Sequential(
            LinearWithLoRA(original_ffn.ffnet[0]),
            original_ffn.ffnet[1],
            LinearWithLoRA(original_ffn.ffnet[2]),
            original_ffn.ffnet[3]
        )

    def forward(self, x):
        return self.ffnet(x)


class PEFTModel(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained_model = pretrained_model
        for i, block in enumerate(self.pretrained_model.transformer_blocks):
            self.pretrained_model.transformer_blocks[i].self_attn = MultiHeadAttentionWithLoRA(block.self_attn)
            
            # according to the paper , in Section 4.2 LoRA is only applied on the Attention heads. However, 
            # recent and common practices involve to apply it also to MLP layers, if you want to do so, uncomment this line:
            # self.pretrained_model.transformer_blocks[i].feed_forward = FeedForwardWithLoRA(block.feed_forward)

        self.freeze_non_lora_layers(self.pretrained_model)
        
    def forward(self, index, targets=None, mask=None):
        logits, loss = self.pretrained_model(index, targets, mask)
        return logits, loss

    def freeze_non_lora_layers(self, module):
        for name, param in module.named_parameters():

            # in original paper, these layers are frozen for both for simplicity
            # and parameter-efficiency. However, as this is a very small example, 
            # training the fully-connected layers was beneficial.

            if "lora" not in name and ".feed_forward.ffnet." not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True


class ModelTrainer:
    def __init__(self, data_handler, model):
        self.data_handler = data_handler
        self.model = model
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=training_params.learning_rate,
            weight_decay=0.02
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        percentage_trainable = (trainable_params / total_params) * 100

        log_message = (f"Total parameters: {total_params}, "
                    f"Trainable parameters: {trainable_params}, "
                    f"Percentage of trainable parameters: {percentage_trainable:.2f}%")

        logging.info(log_message)
        tracking.set_custom_info(
            custom_info=[
                MetricCustomInfo(name="Parameters", value=str(log_message))
                ]
            ) 

    def train(self):
        for iteration in range(training_params.iteration_limit):
            if iteration % training_params.eval_frequency == 0 or iteration == training_params.iteration_limit - 1:
                logging.info('Epoch {} started'.format(iteration))
                losses = self.data_handler.get_estimated_loss(self.model)
                evaluation_msg = 'EPOCH {} | LOSS: Train {:.4f} Valid {:.4f}'.format(
                    str(iteration).ljust(5), losses['train'], losses['val'])
                
                logging.info(evaluation_msg)
                tracking.set_custom_info(
                    custom_info=[MetricCustomInfo(name="Epoch Status", value=str(evaluation_msg))]
                )
                tracking.log_metrics(
                    metrics=[
                        Metric(name="Training Loss", value=float('{:.4f}'.format(losses['train'])), timestamp=datetime.now(timezone.utc), step=iteration),
                        Metric(name="Validation Loss", value=float('{:.4f}'.format(losses['val'])), timestamp=datetime.now(timezone.utc), step=iteration),
                    ]
                )

            batches_x, batches_y, mask = self.data_handler.get_batch('train')
            print(mask)
            logging.info('Sent to Data Handler for Tokenization and Generating Batches for iteration {}'.format(iteration))
            logits, loss = self.model(batches_x, batches_y, mask)
            logging.info('Forward Pass for iteration {}'.format(iteration))

            # Ensure loss requires gradient
            loss = loss.requires_grad_()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            logging.info('Gradient Norm {}'.format(norm))
            logging.info('Backward Pass for iteration {}'.format(iteration))
            self.optimizer.step()
            logging.info('Optimization Step for iteration {}'.format(iteration))
