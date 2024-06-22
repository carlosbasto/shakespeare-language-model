
import torch
import torch.nn as nn
from torch.nn import functional as F
from ShakespeareanGenerator.logger import Logger
from ShakespeareanGenerator.parameters import TrainingParameters
from ai_core_sdk.tracking import Tracking
from ai_core_sdk.models import Metric, MetricCustomInfo
from datetime import datetime, timezone

# start tracking
tracking = Tracking()
logging = Logger()
training_params = TrainingParameters()

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(training_params.embedding_dim, head_size, bias=False)
        self.query = nn.Linear(training_params.embedding_dim, head_size, bias=False)
        self.value = nn.Linear(training_params.embedding_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(training_params.context_length, training_params.context_length)))

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
        out = weights @ v
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(head_size * num_heads, training_params.embedding_dim)
        self.dropout = nn.Dropout(training_params.dropout)

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        out = torch.cat(head_outputs, dim=-1)
        out = self.dropout(self.projection(out))
        return out

class FeedForward(nn.Module):

    def __init__(self, embedding_dim):

        super().__init__()
        self.ffnet = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(training_params.dropout)
        )

    def forward(self, x):
        return self.ffnet(x)

class TransformerBlock(nn.Module):

    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        head_size = embedding_dim // num_heads
        self.self_attn = MultiHeadAttention(num_heads, head_size)
        self.feed_forward = FeedForward(embedding_dim)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        attention_output = x + self.self_attn(self.layer_norm1(x))
        output = attention_output + self.feed_forward(self.layer_norm2(attention_output))

        return output

class ShakespeareanLanguagelModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.embeddings = nn.Embedding(training_params.dictionary_size, training_params.embedding_dim)
        self.position_embeddings = nn.Embedding(training_params.context_length, training_params.embedding_dim)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(training_params.embedding_dim, num_heads=training_params.attention_heads) for _ in range(training_params.num_layers)]
        )
        self.layer_norm = nn.LayerNorm(training_params.embedding_dim)
        self.output = nn.Linear(training_params.embedding_dim, training_params.dictionary_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, index, targets=None):

        B, T = index.shape
        token_embeddings = self.embeddings(index)
        position_embeddings = self.position_embeddings(torch.arange(T, device=index.device))
        x = token_embeddings + position_embeddings
        x = self.transformer_blocks(x)
        x = self.layer_norm(x)
        logits = self.output(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

class ModelTrainer:

    def __init__(self, data_handler, model):
        self.data_handler = data_handler
        self.model = model

        learning_parameters = sum(p.numel() for p in model.parameters()) / 1e6
        msg_to_log = 'The model is learning {} million parameters.'.format(learning_parameters)
        logging.info(msg_to_log)
        msg_to_metrics = '{} million parameters.'.format(learning_parameters)
        tracking.set_custom_info(
            custom_info=[
                MetricCustomInfo(name="Number of Parameters", value=str(msg_to_metrics))
            ]
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=training_params.learning_rate
        )

    def train(self):
        try:
            for iteration in range(training_params.iteration_limit):
                if iteration % training_params.eval_frequency == 0 or iteration == training_params.iteration_limit - 1:
                    logging.info('Epoch {} started'.format(iteration))

                    losses = self.data_handler.get_estimated_loss(self.model)

                    evaluation_msg = 'EPOCH {} | LOSS: Train {:.4f} Valid {:.4f}'.format(
                        str(iteration).ljust(5), losses['train'], losses['val']
                    )
                    logging.info(evaluation_msg)
                    tracking.set_custom_info(
                        custom_info=[
                            MetricCustomInfo(name="Epoch Status", value=str(evaluation_msg))
                        ]
                    )
                    # Metric Logging: Step Information
                    training_loss_msg = '{:.4f}'.format(losses['train'])
                    validation_loss_msg = '{:.4f}'.format(losses['val'])
                    tracking.log_metrics(
                        metrics=[
                            Metric(
                                name="Training Loss",
                                value=float(training_loss_msg),
                                timestamp=datetime.now(timezone.utc),
                                step=iteration
                            ),
                            Metric(
                                name="Validation Loss",
                                value=float(validation_loss_msg),
                                timestamp=datetime.now(timezone.utc),
                                step=iteration
                            ),
                        ]
                    )
                batches_x, batches_y = self.data_handler.get_batch('train')
                logging.info(f'Sent to Data Handler for Tokenization and Generating Batches for iteration {iteration}')
                logits, loss = self.model(batches_x, batches_y)
                logging.info(f'Forward Pass for iteration {iteration}')
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                logging.info(f'Backward Pass for iteration {iteration}')
                self.optimizer.step()
                logging.info(f'Optimization Step for iteration {iteration}')
        except Exception as e:
            logging.error(f'Training failed at iteration {iteration} with error: {e}')
            raise

