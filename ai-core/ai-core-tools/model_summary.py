import torch
from torchinfo import summary
from ShakespeareanGenerator.model.language_models import ShakespeareanLanguagelModel
from ShakespeareanGenerator.parameters import TrainingParameters

# Initialize the training parameters
training_params = TrainingParameters()

# Create an instance of the model
model = ShakespeareanLanguagelModel()

# Move the model to the appropriate device (CPU or GPU)
model.to(training_params.device)

# Define input size and create an input tensor with the correct type (LongTensor)
input_size = (64, training_params.context_length)  # Example input size, modify as necessary
input_tensor = torch.randint(0, training_params.dictionary_size, input_size, dtype=torch.long).to(training_params.device)

# Print the model summary
summary(model, input_data=input_tensor)
