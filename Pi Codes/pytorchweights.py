import torch
from transformers import VisionEncoderDecoderModel

# Load the model
model_name = 'microsoft/trocr-small-printed'
trained_model = VisionEncoderDecoderModel.from_pretrained('checkpoint-' + str(7540))

# Access the model's state_dict, which contains all the weights and parameters
state_dict = trained_model.state_dict()

# Iterate through the state_dict to print out layer names and corresponding weights
for layer_name, param_tensor in state_dict.items():
    print(f"Layer: {layer_name} | Shape: {param_tensor.shape}")
    print(param_tensor)  # This will print the actual weight values
