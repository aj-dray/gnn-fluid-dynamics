

# === LIBRARY IMPORTS ===


import torch
import importlib
import pprint
import os


# === MODULE IMPORTS ===


from .config import Config


# === FUNCTIONS ===


def merge_checkpoint_config(current_config, checkpoint):
    """
    Merge checkpoint config with current config, with current settings overriding old ones.

    Args:
        current_config: Config object from current config file
        checkpoint: Checkpoint dictionary containing 'config' field

    Returns:
        Config: Merged configuration object
    """
    if 'config' not in checkpoint:
        print("Warning: No config found in checkpoint, using current config only")
        return current_config

    old_config_data = checkpoint['config']
    old_config = Config.from_dict(old_config_data)

    print("Merging checkpoint config with current config...")

    # Create merged config data by starting with old config
    merged_data = old_config.to_json(exclude_none=False)

    # Override with current config values (only non-None values)
    current_data = current_config.to_json(exclude_none=True)

    def deep_merge_dict(base_dict, override_dict):
        """Deep merge override_dict into base_dict"""
        for key, value in override_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                deep_merge_dict(base_dict[key], value)
            else:
                base_dict[key] = value

    deep_merge_dict(merged_data, current_data)
    merged_config = Config.from_dict(merged_data)
    
    # Ensure file paths use the correct machine prefix after merging
    if merged_config.settings.machine:
        from .config import MACHINE_PATHS
        current_machine_prefix = MACHINE_PATHS[merged_config.settings.machine]
        
        def fix_absolute_path(path_value, path_name):
            """Fix absolute paths that don't match current machine prefix"""
            if path_value and os.path.isabs(path_value):
                if not path_value.startswith(current_machine_prefix):
                    # Try to find a common suffix by checking against all machine prefixes
                    relative_path = None
                    for machine_name, machine_prefix in MACHINE_PATHS.items():
                        if path_value.startswith(machine_prefix):
                            relative_path = os.path.relpath(path_value, machine_prefix)
                            break
                    
                    if relative_path:
                        # Reconstruct path with current machine prefix
                        new_path = os.path.join(current_machine_prefix, relative_path)
                        print(f"✓ {path_name} updated to use {merged_config.settings.machine} machine prefix: {new_path}")
                        return new_path
            return path_value
        
        # Fix various file paths that might need machine prefix adjustment
        merged_config.dataset.dpath = fix_absolute_path(merged_config.dataset.dpath, "Dataset path")

    
    print("✓ Config merged: checkpoint settings restored, current settings override")

    return merged_config


def backward_compatibility(config):
    # pprint.pprint(config)
    if config.model.name == 'FVGN':
        config.model.name = 'FvgnA'
    if config.model.module == "fvgn":
        config.model.module = "Fvgn"
    # if config.model.grad_weights_order:
    #     config.model.cell_grad_weights_order = config.model.grad_weights_order
    # if not hasattr(config.training, 'dropout_rate'):
    config.training.dropout_rate = 0.0
    return config


def load_model_state_dict_flexible(model, checkpoint_state_dict, strict=False):
    """
    Load state dict into model with flexible key matching.
    Handles missing keys and shape mismatches gracefully.
    
    Args:
        model: PyTorch model to load state into
        checkpoint_state_dict: State dict from checkpoint
        strict: Whether to enforce strict loading (default: False for flexibility)
    
    Returns:
        None
    """
    model_state_dict = model.state_dict()
    
    # Filter checkpoint keys to only include those that exist in the current model
    filtered_checkpoint_state_dict = {}
    missing_keys = []
    unexpected_keys = []
    
    for key, value in checkpoint_state_dict.items():
        if key in model_state_dict:
            # Check if tensor shapes match
            if model_state_dict[key].shape == value.shape:
                filtered_checkpoint_state_dict[key] = value
            else:
                print(f"Warning: Shape mismatch for key '{key}': "
                      f"model expects {model_state_dict[key].shape}, "
                      f"checkpoint has {value.shape}. Skipping.")
                missing_keys.append(key)
        else:
            unexpected_keys.append(key)
    
    # Check for keys that exist in model but not in checkpoint
    for key in model_state_dict:
        if key not in checkpoint_state_dict:
            missing_keys.append(key)
    
    if missing_keys:
        print(f"Warning: Missing keys in checkpoint (will use model's initialized values): {missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}")
    
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint (will be ignored): {unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}")
    
    # Load the filtered state dict with strict=False to allow missing keys
    model.load_state_dict(filtered_checkpoint_state_dict, strict=strict)


def initialise_model(config, checkpoint, dataset):
    """
    Load a model from a checkpoint with all necessary components.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on

    Returns:
        tuple: (model, config) where model is ready for inference
    """
    config = backward_compatibility(config)
    model_module = importlib.import_module(config.model.module)
    model_class = getattr(model_module, config.model.name)

    stats = checkpoint['stats']
    model = model_class(config, None, dataset, stats)

    # Load state dict with flexible key matching
    load_model_state_dict_flexible(model, checkpoint['model_state_dict'])
    model.eval()
    model.to(config.settings.device)

    return model

def update_config(config, train_config):
    if hasattr(train_config.model, 'cell_grad_weights_order') and train_config.model.cell_grad_weights_order:
        config.model.cell_grad_weights_order = train_config.model.cell_grad_weights_order
    if hasattr(train_config.model, 'face_grad_weights_order') and train_config.model.face_grad_weights_order:
        config.model.face_grad_weights_order = train_config.model.face_grad_weights_order
    return config
