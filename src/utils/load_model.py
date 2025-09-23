import torch
import json
import sys
import os
import importlib

# Add the parent directory of 'src' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.config import Parameters
from src.utils.model_loading import load_model_state_dict_flexible
from torch.nn import MSELoss

'''
    Note can reconstruct model from saved checkpoint. \
    Dataset stats are needed for model initialization (but typically saved in checkpoint).
'''
def load_trained_model(model_path, device):
    ## Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    params = Parameters()
    params.__dict__.update(checkpoint['parameters'])  # Update the Parameters object with the dictionary

    # Get model information from the checkpoint or parameters
    model_class_name = params.model["class_name"]
    model_module_name = params.model["class_module"]

    # Load the model class dynamically
    model_module = importlib.import_module(model_module_name)
    model_class = getattr(model_module, model_class_name)

    # Define MSE loss function per element
    def MSE_per_element(pred, target):
        return ((pred - target)**2).mean(dim=0)

    # Get input sizes from checkpoint or use defaults
    node_input_size = checkpoint.get('node_input_size', 13)
    edge_input_size = checkpoint.get('edge_input_size', 3)

    ## Initialize model with the same architecture as during training
    model = model_class(
        params,
        MSE_per_element,
        node_input_size=node_input_size,
        edge_input_size=edge_input_size
    )

    # Load the trained weights
    load_model_state_dict_flexible(model, checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    return model
