# === LIBRARIES ===


from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import Optional, List
import argparse
import json
import os


# === CONSTANTS ===


MACHINE_PATHS = {
    "mac": "/Users/adamdray/Code/data",
    "csd3": "/home/ajd246/rds/rds-t2-cs181-iLDMbuOsGy8/ajd246",
    "csc": "/local/data/public/ajd246/data"
}


# === CLASSES ===


class TrainingConfig(BaseModel):
    """Configuration for training parameters including data, optimization, and regularization."""
    data_subset: Optional[str] = Field(None, description="Dataset subset: train, valid or test")
    data_sim_limit: Optional[int] = Field(None, gt=0, description="Number of simulations")
    data_timestep_range: Optional[List[int]] = Field(None, description="Starting timestep range [a, b]")
    epochs: Optional[int] = Field(None, gt=0, description="Number of training epochs")
    batch_size: Optional[int] = Field(None, gt=0, description="Batch size for training")
    batch_length: Optional[int] = Field(1, gt=0, description="number of consecutive timesteps per batch")
    mini_epoch_size: Optional[int] = Field(None, gt=0, description="Definition of a mini_epoch in terms of samples")
    lr_max: Optional[float] = Field(None, gt=0, description="Maximum learning rate")
    lr_min: Optional[float] = Field(None, ge=0, description="Minimum learning rate")
    lr_class: Optional[str] = Field(None, description="Learning rate scheduler class")
    lr_wu: Optional[float] = Field(None, ge=0, le=1, description="Warmup percentage distance")
    lr_wu_gamma: Optional[float] = Field(None, ge=0, le=1, description="Warmup ratio")
    lr_ms1: Optional[float] = Field(None, gt=0, description="LR scheduler milestone 1")
    lr_ms1_gamma: Optional[float] = Field(None, gt=0, le=1, description="LR gamma for milestone 1")
    lr_ms2: Optional[float] = Field(None, gt=0, description="LR scheduler milestone 2")
    lr_ms2_gamma: Optional[float] = Field(None, gt=0, le=1, description="LR gamma for milestone 2")
    lr_ms3: Optional[float] = Field(None, gt=0, description="LR scheduler milestone 3")
    optimizer_name: Optional[str] = Field(None, description="Optimizer name (Adam, AdamW, etc.)")
    weight_decay: Optional[float] = Field(None, ge=0, description="Weight decay for regularization")
    clip_grad_norm: Optional[float] = Field(None, ge=0, description="Gradient clipping norm")
    dropout_rate: Optional[float] = Field(None, ge=0, le=1, description="Dropout rate")
    noise_std_norm: Optional[float] = Field(None, ge=0, description="Standard deviation of noise for normalisation")
    noise_std: Optional[float] = Field(None, ge=0, description="Standard deviation of noise")
    loss_weights: Optional[dict] = Field(None, description="Weights for different loss components (dict for JSON compatibility)")
    num_workers: Optional[int] = Field(None, ge=0, description="Number of data loader workers")
    persistent_workers: Optional[bool] = Field(None, description="Use persistent workers for data loading")
    prefetch_factor: Optional[int] = Field(None, ge=0, description="Number of batches to prefetch")
    pushforward_factor: Optional[int] = Field(None, description="Use pushforward")

    @model_validator(mode='after')
    def validate_mini_epoch_size_batch_size_relationship(self):
        """Validate that mini_epoch_size is a multiple of batch_size when both are provided."""
        if self.mini_epoch_size is not None and self.batch_size is not None:
            if self.mini_epoch_size % self.batch_size != 0:
                raise ValueError(
                    f"mini_epoch_size ({self.mini_epoch_size}) must be a multiple of batch_size ({self.batch_size})"
                )
        return self


class LoggingConfig(BaseModel):
    project: Optional[str] = Field(None, description="Indication of dataset being used")
    group: Optional[str] = Field(None, description="Experiment group name")
    name: Optional[str] = Field(None, description="Run name")
    notes: Optional[str] = Field(None, description="Run notes or description")
    run_count: Optional[int] = Field(0, description="Number of training runs on this particular model")
    loss_frequency: Optional[int] = Field(None, gt=0, description="Frequency of validation per training round")
    valid_frequency: Optional[int] = Field(None, gt=0, description="Frequency of validation per training rounds")
    save_frequency: Optional[int] = Field(None, ge=0, description="Frequenc of validation per training save model")
    save_overwrite: Optional[bool] = Field(None, description="Whether best new checkpoint replaces old")
    use_wandb: Optional[bool] = Field(True, description="Whether to use wandb")
    use_tensorboard: Optional[bool] = Field(False)
    is_debug: Optional[bool] = Field(None, description="Whether to run in debug mode")


class DatasetConfig(BaseModel):
    module: Optional[str] = Field(None, description="Dataset class module path")
    name: Optional[str] = Field(None, description="Dataset class name")
    dpath: Optional[str] = Field(None, description="Path to dataset files")
    stats_fpath: Optional[str] = Field(None, description="Statistics file path")
    shuffle: Optional[bool] = Field(None, description="Shuffle dataset")
    dt: Optional[float] = Field(None, description="Time step size")
    stats_recompute: Optional[bool] = Field(None, description="Whether to accumulate statistics")
    grad_weights_recompute: Optional[bool] = Field(None, description="Whether to recompute gradient weights")


class RolloutConfig(BaseModel):
    data_subset: Optional[str] = Field(None, description="Dataset subset: train, valid or test")
    data_sim_limit: Optional[int] = Field(None, gt=0, description="Number of simulations")
    data_timestep_range: Optional[List[int]] = Field(None, description="Starting timestep range [a, b]")
    data_sim_index: Optional[List[int]] = Field(None, description="List of trajector indicies to use")
    batch_size: Optional[int] = Field(None, gt=0)
    save_frequency: Optional[int] = Field(1, gt=0, description="Frequency of saving each timestep")
    num_workers: Optional[int] = Field(None, ge=0, description="Number of data loader workers")
    prefetch_factor: Optional[int] = Field(None, ge=0, description="Number of batches to prefetch")
    persistent_workers: Optional[bool] = Field(None, description="Use persistent workers for data loading")
    loss_frequency: Optional[int] = Field(None, gt=0, description="Frequency of loss computation")
    snapshot_indices: Optional[List[int]] = Field(None, description="List of snapshot indices to save")

    @model_validator(mode='after')
    def validate_data_sim_index_length(self):
        """Check that len(data_sim_index) == data_sim_limit if both are provided."""
        if self.data_sim_index is not None and self.data_sim_limit is not None:
            if len(self.data_sim_index) != self.data_sim_limit:
                raise ValueError(
                    f"Length of data_sim_index ({len(self.data_sim_index)}) must equal data_sim_limit ({self.data_sim_limit})"
                )
        return self


class SettingsConfig(BaseModel):
    machine: Optional[str] = Field(None, description="Machine type (mac, csd3, csc)")
    device: Optional[str] = Field(None, description="Device type (cpu, cuda, mps)")
    multi_gpu: Optional[bool] = Field(None, description="Use multiple GPUs")
    num_gpus: Optional[int] = Field(None, gt=0, description="Number of GPUs to use")
    pin_memory: Optional[bool] = Field(None, description="Pin memory for data loading")
    random_seed: Optional[int] = Field(0, description="Random seed for reproducibility")


class ModelConfig(BaseModel):
    module: Optional[str] = Field(None, description="Model class module path")
    name: Optional[str] = Field(None, description="Model class name")
    hidden_width: Optional[int] = Field(None, gt=0, description="Hidden layer width")
    mp_num: Optional[int] = Field(None, gt=0, description="Message passing iterations")
    fpath: Optional[str] = Field(None, description="Model file path")
    cell_grad_weights_use: Optional[bool] = Field(None, description="Use gradient weights")
    cell_grad_weights_order: Optional[int] = Field(None, description="Gradient weights poly order")
    face_grad_weights_use: Optional[bool] = Field(None, description="Use gradient weights for faces")
    face_grad_weights_order: Optional[int] = Field(None, description="Gradient weights poly order")
    timestep_stride: Optional[int] = Field(1, gt=0, description="Stride for timesteps in data loading")
    bundle_size: Optional[int] = Field(None, description="Bundle size for temporal bundling")


class PreprocConfig(BaseModel):
    data_subset: Optional[str] = Field(None, description="Data subset")
    data_sim_limit: Optional[int] = Field(None, gt=0, description="Number of simulations")
    data_timestep_range: Optional[List[int]] = Field(None, description="Starting timestep range [a, b]")
    h5: Optional[bool] = Field(None)
    h5_fpath: Optional[str] = Field(None, description="HDF5 file path")
    stats: Optional[bool] = Field(None)
    stats_fpath: Optional[str] = Field(None, description="Statistics file path")
    num_workers: Optional[int] = Field(None, description="Number of workers for data loading")
    batch_size: Optional[int] = Field(None, description="Batch size for data loading")


class Config(BaseModel):
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    settings: SettingsConfig = Field(default_factory=SettingsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    preproc: PreprocConfig = Field(default_factory=PreprocConfig)

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True
    )

    def __init__(self, **data):
        super().__init__(**data)

    def set_from_file(self, config):
        for section_name in ["training", "logging", "settings", "dataset", "model", "validation", "preproc"]:
            if section_name in config:
                section_data = config[section_name]
                section_obj = getattr(self, section_name)

                for key, value in section_data.items():
                    if hasattr(section_obj, key):
                        # Special handling for dataset dpath
                        if section_name == "dataset" and key == "dpath" and value is not None:
                            if self.settings.machine and not os.path.isabs(value):
                                data_path = MACHINE_PATHS[self.settings.machine]
                                value = os.path.join(data_path, value)
                        setattr(section_obj, key, value)

    def append_data_path(self, path):
        if self.settings.machine:
            data_path = MACHINE_PATHS[self.settings.machine]
            return os.path.join(data_path, path)
        return path

    @classmethod
    def from_dict(cls, data: dict) -> 'Config':
        """Load configuration from JSON file."""

        config = cls(**data)

        # Post-process to resolve data paths
        if (config.settings.machine and
            config.dataset.dpath and
            not os.path.isabs(config.dataset.dpath)):
            data_path = MACHINE_PATHS[config.settings.machine]
            config.dataset.dpath = os.path.join(data_path, config.dataset.dpath)

        return config

    def to_json(self, exclude_none: bool = True) -> dict:
        """Return configuration as a JSON-serializable dict."""
        data = self.model_dump(exclude_none=exclude_none)
        return data

    def to_flat_json(self, exclude_none: bool = True) -> dict:
        """Return configuration as a flattened JSON-serializable dict (level1.level2.name format)."""
        data = self.model_dump(exclude_none=exclude_none)
        flat_data = {}

        def flatten_dict(d, parent_key=''):
            for key, value in d.items():
                new_key = f"{parent_key}.{key}" if parent_key else key
                if isinstance(value, dict):
                    flatten_dict(value, new_key)
                else:
                    flat_data[new_key] = value

        flatten_dict(data)
        return flat_data



# === FUNCTIONS ===


def load_config_file(filepath):
    with open(filepath, 'r') as f:
        config_file = json.load(f)
    return config_file

def file_parser(config_file, parameters):
    parameters.set_from_file(config_file)

def argument_parser(parameters):
    parser = argparse.ArgumentParser(description='Flags for training')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode (checkpoints and model won\'t be saved)')
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()

    if args.debug:
        parameters.is_debug = True
        parameters.logging.group = "debug"

    if args.config is not None:
        return args.config
    else:
        raise ValueError("No config file provided. Use --config to specify a config file.")


# === TESTING ===


if __name__ == "__main__":
    # Test loading the config file
    config_path = "config/train-CF.json.json"

    try:
        # Test 1: Load config from JSON
        print("Testing config loading...")
        config = Config.from_json(config_path)
        print("✓ Config loaded successfully!")

        # Test 2: Print some key values to verify
        print("\nKey config values:")
        print(f"Dataset class: {config.dataset.class_name}")
        print(f"Model hidden width: {config.model.hidden_width}")
        print(f"Training epochs: {config.training.epochs}")
        print(f"Device: {config.settings.device}")
        print(f"Loss weights: {config.training.loss_weights}")

        # Test 3: Test saving back to JSON
        print("\nTesting config saving...")
        test_output_path = "test_config_output.json"
        config.to_json(test_output_path)
        print(f"✓ Config saved to {test_output_path}")

        # Test 4: Test flattened JSON output
        flat_output_path = "test_config_flat.json"
        flat_j = config.to_flat_json(flat_output_path)
        # print(flat_j)
        print(f"✓ Flattened config saved to {flat_output_path}")

        # Test 5: Test data path appending
        test_path = config.append_data_path("test/path")
        print(f"Data path test: {test_path}")

        print("\nAll tests passed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
