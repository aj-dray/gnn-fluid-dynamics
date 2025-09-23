# === LIBRARIES ===


from abc import ABC, abstractmethod
import torch
from utils.normalisation import CustomNormalizer


# === UTILITY FUNCTIONS ===


def build_mlp(config, in_size, hidden_size, out_size, norm_layer=True):
    """
    Build a standard MLP with SiLU activation and optional dropout/normalization.
    
    Args:
        config: Configuration object with training parameters
        in_size: Input feature dimension
        hidden_size: Hidden layer dimension
        out_size: Output feature dimension
        norm_layer: Whether to apply LayerNorm to output
        
    Returns:
        torch.nn.Module: MLP network
    """
    layers = []
    layers.append(torch.nn.Linear(in_size, hidden_size))
    layers.append(torch.nn.SiLU())
    if hasattr(config, "training") and hasattr(config.training, "dropout_rate") and config.training.dropout_rate > 0:
        layers.append(torch.nn.Dropout(p=config.training.dropout_rate))
    layers.append(torch.nn.Linear(hidden_size, hidden_size))
    layers.append(torch.nn.SiLU())
    if hasattr(config, "training") and hasattr(config.training, "dropout_rate") and config.training.dropout_rate > 0:
        layers.append(torch.nn.Dropout(p=config.training.dropout_rate))
    layers.append(torch.nn.Linear(hidden_size, out_size))

    module = torch.nn.Sequential(*layers)
    if norm_layer:
        return torch.nn.Sequential(module, torch.nn.LayerNorm(normalized_shape=out_size))
    return module


# === BASE MODEL CLASS ===


class Model(torch.nn.Module, ABC):
    """
    Abstract base class for all neural network models in the project.
    Provides common functionality for normalization, feature handling, and loss computation.
    All concrete models must implement abstract methods for features, normalization, forward pass, and loss.
    """
    # Defaults Settings
    cell_grad_weights_use = False
    face_grad_weights_use = False
    pushforward_use = False

    def __init__(self, config, loss_func, dataset, stats):
        """
        Initialize the model with configuration and data-dependent parameters.
        
        Args:
            config: Configuration object with model parameters
            loss_func: Loss function for training
            dataset: Dataset instance for feature size determination
            stats: Dataset statistics for normalization setup
        """
        super().__init__()
        self.config = config
        self.loss_func = loss_func
        self.hidden_size = config.model.hidden_width
        self.input_sizes, self.output_sizes = self.get_feature_sizes(dataset)
        self.set_normalizer(stats)


    @classmethod
    @abstractmethod
    def get_feature_sizes(cls, dataset):
        """
        Get input and output feature dimensions based on dataset characteristics.

        Returns:
            tuple: (input_sizes, output_sizes) where each is a list of dimensions
        """
        pass

    @classmethod
    @abstractmethod
    def get_normalisation_map(cls):
        """
        Define normalization specifications for inputs and outputs.

        Returns:
            tuple: (registry, inputs, outputs) for normalization setup
        """
        pass

    @abstractmethod
    def forward(self, graphs, mode='train'):
        """
        Forward pass through the model.

        Args:
            graphs: Input graph data structures
            mode: Either 'train' or 'rollout' - determines normalization behavior

        Returns:
            dict: Model predictions with appropriate keys
        """
        pass

    @abstractmethod
    def loss(self, output, graphs):
        """
        Compute loss between model output and ground truth.

        Args:
            output: Model predictions
            graphs: Ground truth data

        Returns:
            dict: Loss components and total loss
        """
        pass

    def set_normalizer(self, stats):
        """
        Set up data normalization using provided statistics.

        Args:
            stats: Dataset statistics for normalization
        """
        registry, inputs, outputs = self.get_normalisation_map()
        self.normalizer = CustomNormalizer(stats, registry, inputs, outputs)

    def transform_features(self, graphs):
        """
        Apply feature transformations to input graphs.

        Args:
            graphs: Input graph data

        Returns:
            Transformed graph data
        """
        pass

    def update_features(self, graphs):
        """
        Update or modify graph features during processing.

        Args:
            graphs: Graph data to update

        Returns:
            Updated graph data
        """
        pass

    def count_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
