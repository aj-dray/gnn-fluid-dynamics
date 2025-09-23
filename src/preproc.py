"""
Preprocessing script for dataset preparation.
Exact conversion depends on DataSet class used.
Statistic accumulation now redundant as this is handled within the training script.
"""


# === LIBRARIES ===


import time
import os
import torch
from torch_geometric.loader import DataLoader
import json
from importlib import import_module
import argparse
from torch.utils.data import SequentialSampler, BatchSampler


# === MODULES ===


from utils.config import Config


# === FUNCTIONS ===


def calculate_dataset_statistics(dataloader, tensor_getters, save_path):
    """
    Calculate mean, std, min, max statistics for datasets.
    DEPRECATED - statistics are now handled within the training script.
    
    Args:
        dataloader: PyTorch DataLoader for the dataset
        tensor_getters: Dict mapping stat names to tensor extraction functions
        save_path: File path to save computed statistics
    """

    # Initialize accumulators
    sums = {}
    squared_sums = {}
    counts = {}
    max_vals = {}
    min_vals = {}

    # Use mixed precision for efficiency but accuracy
    accum_dtype = torch.float64

    # Process in chunks to avoid memory issues
    start_time = time.time()
    total_batches = len(dataloader)
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Print progress
            elapsed_time = time.time() - start_time
            print(f"\r{batch_idx+1} / {total_batches}; t={elapsed_time:.2f}s", end='', flush=True)

            # Process each tensor type
            for name, getter in tensor_getters.items():
                tensor = getter(batch)
                if tensor is None or tensor.numel() == 0:
                    continue

                # Initialize accumulators if needed
                if name not in sums:
                    sums[name] = torch.zeros(tensor.shape[1:], dtype=accum_dtype)
                    squared_sums[name] = torch.zeros(tensor.shape[1:], dtype=accum_dtype)
                    counts[name] = 0
                    max_vals[name] = torch.full(tensor.shape[1:], float('-inf'), dtype=accum_dtype)
                    min_vals[name] = torch.full(tensor.shape[1:], float('inf'), dtype=accum_dtype)

                # Convert and accumulate statistics in high precision
                # Use in-place operations where possible
                tensor_fp = tensor.to(accum_dtype)
                sums[name].add_(tensor_fp.sum(dim=0))
                squared_sums[name].add_((tensor_fp ** 2).sum(dim=0))
                counts[name] += tensor.shape[0]

                # Update min/max
                max_vals[name] = torch.maximum(max_vals[name], tensor_fp.max(dim=0)[0])
                min_vals[name] = torch.minimum(min_vals[name], tensor_fp.min(dim=0)[0])

                # Free memory explicitly
                del tensor_fp

    # Final calculations
    print("\n")
    stats = {}
    calculation_device = "cpu"

    for name in sums:
        mean = (sums[name] / counts[name]).to(calculation_device)
        var = torch.clamp((squared_sums[name] / counts[name]).to(calculation_device) - mean**2, min=0.0)
        std = torch.sqrt(var)

        # Convert to Python lists for JSON serialization
        stats[f'{name}_mean'] = mean.cpu().tolist()
        stats[f'{name}_std'] = std.cpu().tolist()
        stats[f'{name}_min'] = min_vals[name].cpu().tolist()
        stats[f'{name}_max'] = max_vals[name].cpu().tolist()

    # Save statistics
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=4, sort_keys=True)
        print("Statistics saved to", save_path)

    return 0


def create_proc_file(dataset, save_path):
    """
    Process and save dataset to HDF5 format.
    
    Args:
        dataset: Dataset instance to process
        save_path: Output file path for processed data
    """
    start = time.time()
    save_path = config.append_data_path(save_path)
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dataset.preprocess(save_path)
    elapsed_time = time.time() - start
    print(f"Processed file saved in {elapsed_time} to", save_path)
    return

# === MAIN ===

if __name__ == '__main__':
    # CL arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    ## Config
    with open(args.config, 'r') as f:
        data = json.load(f)
    config = Config.from_dict(data)

    # Import classes and modules
    dataset_module = import_module(config.dataset.module)
    dataset_class = getattr(dataset_module, config.dataset.name)

    # File paths
    data_directory = config.append_data_path(config.dataset.dpath)
    h5_save_path = config.preproc.h5_fpath

    if config.preproc.h5:
        for subset in ['valid', 'test']:
            # Set the data subset for this iteration
            config.preproc.data_subset = subset
            train_dataset = dataset_class(data_directory, config, 'preproc', noise=False)

            create_proc_file(train_dataset, os.path.join(h5_save_path, f'{subset}.h5'))

    if config.preproc.stats: #REDUNDANT
        data_directory = h5_save_path
        train_dataset = dataset_class(data_directory, config, mode='preproc', noise=False, shuffle=False)
        train_sampler = BatchSampler(SequentialSampler(train_dataset), batch_size=config.preproc.batch_size, drop_last=False)
        train_loader = DataLoader(train_dataset, pin_memory=False, num_workers=config.preproc.num_workers)

        # for FVGN
        tensor_getters={
                'cell_x': lambda b: b[0].x,
                'cell_y': lambda b: b[0].y,
                'face_x': lambda b: b[1].x,
                'face_y': lambda b: b[1].y,
        }
        stats_save_path = config.preproc.stats_fpath
        calculate_dataset_statistics(train_loader, tensor_getters, stats_save_path)
