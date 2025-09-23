import torch
import torch.nn as nn
import numpy as np
import json
import os
from torch.utils.data import DataChunk
import tqdm


class CustomAccumulator():
    def __init__(self, registry, input_map, output_map, device, stats_fpath=None):
        self.registry = registry
        self.input_map = input_map
        self.output_map = output_map
        self.stats = {}
        self.final_stats = {}
        self.device = device
        self.stats_filepath = stats_fpath

    def _load_existing_stats(self):
        """Load existing stats from file if it exists, always return a dict."""
        print(f"\tLoading existing stats from {self.stats_filepath}")
        if self.stats_filepath and os.path.exists(self.stats_filepath):
            try:
                with open(self.stats_filepath, 'r') as f:
                    result = json.load(f)
                    self.final_stats = result
                    if result is None:
                        return {}
                    return result
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def check_existing(self):
        """Check what is needed from existing"""
        # Load existing stats
        existing_stats = self._load_existing_stats()
        # Determine which keys need accumulation
        required_keys = self._get_required_keys()
        missing_keys = required_keys - set(existing_stats.keys())

        # If no missing keys, return existing stats
        if len(missing_keys) == 0:
            print("\tAll required statistics already exist. Skipping accumulation.")
            return True

        print(f"Computing statistics for missing keys: {missing_keys}")
        return False

    def _get_required_keys(self):
        """Get all unique normalization keys that need stats"""
        required_keys = set()

        # Collect from input map
        for key, (_, norm_key) in self.input_map.items():
            # Only add keys that have extractors in registry (skip None extractors)
            if norm_key in self.registry and self.registry[norm_key][0] is not None:
                required_keys.add(norm_key)

        # Collect from output map
        for key, (_, norm_key) in self.output_map.items():
            # Only add keys that have extractors in registry (skip None extractors)
            if norm_key in self.registry and self.registry[norm_key][0] is not None:
                required_keys.add(norm_key)

        return required_keys

    def _init_stats(self, key):
        """Initialize stats accumulation for a specific field"""
        if key not in self.stats:
            self.stats[key] = {
                'mean': torch.tensor(0.0, device=self.device),
                'M2': torch.tensor(0.0, device=self.device),  # For Welford's algorithm
                'min_val': torch.tensor(float('inf'), device=self.device),
                'max_val': torch.tensor(float('-inf'), device=self.device),
                'count': 0
            }

    def _accumulate_data(self, key, data):
        """Accumulate statistics using numerically stable batch Welford's algorithm"""
        self._init_stats(key)
        # if key == 'face_angle':
        #     print(data)
        # Flatten data for scalar statistics but keep on GPU
        flat_data = data.view(-1).detach()
        # print(f"Accumulating data for key: {key}, shape: {flat_data.shape}")
        # Update min/max using correct tensor keys
        self.stats[key]['min_val'] = torch.min(self.stats[key]['min_val'], torch.min(flat_data))
        self.stats[key]['max_val'] = torch.max(self.stats[key]['max_val'], torch.max(flat_data))

        # Batch Welford's algorithm for numerical stability
        old_count = self.stats[key]['count']
        new_count = old_count + flat_data.numel()

        if old_count == 0:
            # First batch
            self.stats[key]['mean'] = torch.mean(flat_data)
            self.stats[key]['M2'] = torch.sum((flat_data - self.stats[key]['mean']) ** 2)
        else:
            # Update with new batch
            batch_mean = torch.mean(flat_data)
            batch_size = flat_data.numel()

            # Combined mean
            delta = batch_mean - self.stats[key]['mean']
            new_mean = self.stats[key]['mean'] + delta * batch_size / new_count

            # Combined M2 (sum of squared deviations)
            batch_M2 = torch.sum((flat_data - batch_mean) ** 2)
            self.stats[key]['M2'] = (self.stats[key]['M2'] + batch_M2 +
                                   delta ** 2 * old_count * batch_size / new_count)

            self.stats[key]['mean'] = new_mean

        self.stats[key]['count'] = new_count

    def run(self, dataloader, stats_recompute=False):
        # If recompute is requested, wipe stats file and start fresh
                # Determine which keys need accumulation
        required_keys = self._get_required_keys()

        # Load existing stats
        if stats_recompute:
            existing_stats = {}
            missing_keys = required_keys
        else:
            existing_stats = self._load_existing_stats()
            if not existing_stats:
                print("No existing statistics found. Starting fresh.")
                missing_keys = required_keys
            else:
                # Determine which keys need accumulation
                missing_keys = required_keys - set(existing_stats.keys())

        with torch.no_grad():
            for graphs in tqdm.tqdm(dataloader, desc="Accumulating Statistics"):
                # Move graphs to device
                graphs = [graph.to(self.device) for graph in graphs]
                # print(graphs[1].y.shape)
                # Only accumulate stats for missing keys
                for norm_key in missing_keys:
                    if norm_key in self.registry:
                        stats_extractor, _ = self.registry[norm_key]
                        # Skip None extractors (these are for derived statistics)
                        if stats_extractor is not None:
                            data = stats_extractor(graphs)
                            self._accumulate_data(norm_key, data)

        # Compute final statistics for new keys
        final_stats = existing_stats.copy()
        for key, accumulator in self.stats.items():
            count = accumulator['count']
            if count > 1:
                mean = accumulator['mean']
                # Welford's algorithm: variance = M2 / (n-1) for sample variance
                var = accumulator['M2'] / (count - 1)
                std = torch.sqrt(torch.clamp(var, min=1e-16))  # Minimum std threshold
                # print(f"{key}: std = {std}")
                final_stats[key] = {
                    "mean": float(mean.cpu()),
                    "std": float(std.cpu()),
                    "min": float(accumulator['min_val'].cpu()),
                    "max": float(accumulator['max_val'].cpu())
                }
            elif count == 1:
                # Single data point case
                final_stats[key] = {
                    "mean": float(accumulator['mean'].cpu()),
                    "std": 1e-4,  # Default std for single point
                    "min": float(accumulator['min_val'].cpu()),
                    "max": float(accumulator['max_val'].cpu())
                }

        # Compute derived statistics for keys with None extractors
        self._compute_derived_stats(final_stats)

        self.final_stats = final_stats
        # print(final_stats)
        # self.save_stats()
        return final_stats

    def _compute_derived_stats(self, stats_dict):
        """Compute derived statistics for keys with None extractors"""
        # Check if we need to compute characteristic_pressure
        if 'characteristic_pressure' in [norm_key for _, (_, norm_key) in {**self.input_map, **self.output_map}.items()]:
            if 'characteristic_velocity' in stats_dict:
                # characteristic_pressure = 0.5 * characteristic_velocity^2
                v_char_max = stats_dict['characteristic_velocity']['max']
                char_pressure_max = 0.5 * v_char_max ** 2

                stats_dict['characteristic_pressure'] = {
                    "mean": char_pressure_max / 2,  # Reasonable estimate
                    "std": char_pressure_max / 4,   # Reasonable estimate
                    "min": 0.0,
                    "max": char_pressure_max
                }

    def get_stats(self):
        return self.final_stats

    def save_stats(self):
        os.makedirs(os.path.dirname(self.stats_filepath), exist_ok=True)
        with open(self.stats_filepath, 'w') as f:
            json.dump(self.final_stats, f, indent=2)

class CustomNormalizer(nn.Module):
    def __init__(self, stats_dict, registry, input_map, output_map):
        super().__init__()
        self.epsilon = 1e-10
        self.registry = registry

        # Register stats as buffers
        self.stats_keys = list(stats_dict.keys())
        for key, stat_dict in stats_dict.items():
            for stat_name, stat_value in stat_dict.items():
                buffer_name = f"{key}_{stat_name}"
                self.register_buffer(buffer_name, torch.tensor(stat_value, dtype=torch.float))

        # Store norm function lookup
        self.norm_functions = {
            'mean_scale': mean_scale,
            'max_scale': max_scale,
            'z_score': z_score,
            'min_max': min_max,
            'std_scale': std_scale,
        }

        # Wrap the map functions with accessors
        self.input_map = self._wrap_map_with_accessors(input_map, 'input')
        self.output_map = self._wrap_map_with_accessors(output_map, 'output')

    def _accessor(self, func):
        def wrapper(graphs_or_outputs, data=None):
            try:
                target = func(graphs_or_outputs)
            except (TypeError, IndexError):
                # Handle case where graphs_or_outputs contains None values
                return None
            if data is None:
                return target
            else:
                target[:] = data
                return target
        return wrapper

    def _wrap_map_with_accessors(self, map_dict, map_type):
        wrapped = {}
        for key, (extractor, norm_key) in map_dict.items():
            _, norm_func_name = self.registry[norm_key]
            norm_func = self.norm_functions[norm_func_name]
            wrapped[key] = (self._accessor(extractor), norm_func, norm_key)
        return wrapped

    def input(self, graphs, inverse=False):
        for key, (accessor, norm_func, norm_key) in self.input_map.items():
            data = accessor(graphs)
            # Reconstruct stats dict from buffers - include all available stats
            stats = {stat_name: getattr(self, f"{norm_key}_{stat_name}")
                    for stat_name in ['mean', 'std', 'min', 'max'] if hasattr(self, f"{norm_key}_{stat_name}")}
            normalized_data = norm_func(data, stats, inverse)
            accessor(graphs, normalized_data)

        return graphs

    def output(self, outputs, inverse=False):
        for key, (accessor, norm_func, norm_key) in self.output_map.items():
            data = accessor(outputs)
            # Skip normalization if data is None or empty
            if data is None:
                continue
            # Reconstruct stats dict from buffers
            stats = {stat_name: getattr(self, f"{norm_key}_{stat_name}")
                    for stat_name in ['mean', 'std', 'min', 'max'] if hasattr(self, f"{norm_key}_{stat_name}")}
            normalized_data = norm_func(data, stats, inverse)
            accessor(outputs, normalized_data)

        return outputs


def z_score(data, stats, inverse=False):
    epsilon = 1e-8  # Prevent division by zero
    min_std = 1e-8  # Minimum std threshold for numerical stability
    if not inverse:
        std_val = max(stats["std"], min_std)
        return (data - stats["mean"]) / (std_val + epsilon)
    else:
        std_val = max(stats["std"], min_std)
        return data * (std_val + epsilon) + stats["mean"]


def mean_scale(data, stats, inverse=False):
    epsilon = 1e-8  # Prevent division by zero or very small mean
    if not inverse:
        return data / (stats["mean"] + epsilon)
    else:
        return data * (stats["mean"] + epsilon)

def std_scale(data, stats, inverse=False):
    epsilon = 1e-8  # Prevent division by zero or very small mean
    if not inverse:
        return data / (stats["std"] + epsilon)
    else:
        return data * (stats["std"] + epsilon)


def min_max(data, stats, inverse=False):
    epsilon = 1e-8  # Prevent division by zero range
    if not inverse:
        range_val = stats["max"] - stats["min"]
        return (data - stats["min"]) / (range_val + epsilon)
    else:
        range_val = stats["max"] - stats["min"]
        return data * (range_val + epsilon) + stats["min"]


def max_scale(data, stats, inverse=False):
    epsilon = 1e-8  # Prevent division by zero
    if not inverse:
        return data / (stats["max"] + epsilon)
    else:
        return data * (stats["max"] + epsilon)


def normalize_face_area(face_area, cell_volume, edge_index, dt, batchNorm):
    return batchNorm(
        (
            face_area
            * (
                torch.mean(dt)
                / (
                    (
                        torch.index_select(
                            cell_volume, 0, edge_index[0]
                        )
                        + torch.index_select(
                            cell_volume, 0, edge_index[1]
                        )
                    )
                    / 2
                )
            )
        ).view(-1, 1)
    )

def normalize_vol_dt(cell_volume, edge_index, dt, batchNorm):
    return batchNorm(
        (
            1.0
            * (
                torch.mean(dt)
                / (
                    (
                        torch.index_select(
                            cell_volume, 0, edge_index[0]
                        )
                        + torch.index_select(
                            cell_volume, 0, edge_index[1]
                        )
                    )
                    / 2
                )
            )
        ).view(-1, 1)
    )

# class NormalizerZScore(torch.nn.Module):
#     def __init__(self, stats, name, mask=None):
#         super().__init__()
#         self.epsilon = 1e-10
#         self.mask = mask
#         # Register as buffers so they move with the model
#         self.register_buffer('mean', torch.tensor(stats[f"{name}_mean"]))
#         self.register_buffer('std', torch.tensor(stats[f"{name}_std"]))


#     def forward(self, data):
#         mask = self.mask
#         if mask is not None:
#             data[:, mask[0]:mask[1]] = (data[:, mask[0]:mask[1]] - self.mean[mask[0]:mask[1]]) / (self.std[mask[0]:mask[1]] + self.epsilon)
#         else:
#             data = (data - self.mean) / (self.std + self.epsilon)
#         return data

#     def inverse(self, data):
#         mask = self.mask
#         if mask is not None:
#             data[:, mask[0]:mask[1]] = data[:, mask[0]:mask[1]] * (self.std[mask[0]:mask[1]] + self.epsilon) + self.mean[mask[0]:mask[1]]
#         else:
#             data = data * (self.std + self.epsilon) + self.mean
#         return data

# class NormalizerPhysical(torch.nn.Module):
#     def __init__(self, stats, name, mask=None, vectors=None):
#         super().__init__()
#         self.epsilon = 1e-16
#         self.mask = mask
#         # Register as buffer so it moves with the model
#         self.register_buffer('mean', torch.tensor(stats[f"{name}_mean"]))

#     def forward(self, data):
#         mask = self.mask
#         if mask is not None:
#             data[:, mask[0]:mask[1]] = data[:, mask[0]:mask[1]] / (self.mean[mask[0]:mask[1]] + self.epsilon)
#         else:
#             data = data / (self.mean + self.epsilon)
#         return data

#     def inverse(self, data):
#         mask = self.mask
#         if mask is not None:
#             data[:, mask[0]:mask[1]] = data[:, mask[0]:mask[1]] * (self.mean[mask[0]:mask[1]] + self.epsilon)
#         else:
#             data = data * (self.mean + self.epsilon)
#         return data

# class StatsAccumulator:
#     def __init__(self, device='cpu', vectors=None):
#         self.sum = None
#         self.sum_sq = None
#         self.min_val = None
#         self.max_val = None
#         self.count = 0
#         self.device = device
#         self.vectors = vectors if vectors is not None else []
#         # Keep accumulators on GPU if device is GPU for faster computation
#         self.use_gpu = device != 'cpu' and torch.cuda.is_available()

#     def _compute_magnitudes(self, data):
#         """Compute magnitudes for vectors and keep scalars as-is"""
#         if len(self.vectors) == 0:
#             # If no vector info provided, treat all as scalars
#             return data

#         magnitudes = []

#         # Process each component defined by consecutive boundary indices
#         for i in range(len(self.vectors) - 1):
#             start_idx = self.vectors[i]
#             end_idx = self.vectors[i + 1]

#             if end_idx - start_idx == 1:
#                 # Scalar: keep as-is
#                 magnitudes.append(data[:, start_idx])
#             else:
#                 # Vector: compute magnitude
#                 vec_data = data[:, start_idx:end_idx]
#                 mag = torch.norm(vec_data, dim=1)
#                 magnitudes.append(mag)

#         return torch.stack(magnitudes, dim=1)

#     def _expand_stats(self, stats):
#         """Expand magnitude statistics back to original tensor dimensions"""
#         if len(self.vectors) == 0:
#             return stats

#         expanded = []
#         stat_idx = 0

#         # Process each component defined by consecutive boundary indices
#         for i in range(len(self.vectors) - 1):
#             start_idx = self.vectors[i]
#             end_idx = self.vectors[i + 1]
#             component_size = end_idx - start_idx

#             # Repeat the magnitude statistic for each component
#             for _ in range(component_size):
#                 expanded.append(stats[stat_idx])
#             stat_idx += 1

#         return np.array(expanded)

#     def update(self, data):
#         # Ensure data is a tensor
#         if not isinstance(data, torch.Tensor):
#             raise ValueError("Data must be a torch.Tensor")

#         # Move data to computation device if needed
#         if self.use_gpu and data.device.type == 'cpu':
#             data = data.to(self.device)

#         # Compute magnitudes for vectors and keep scalars
#         mag_data = self._compute_magnitudes(data)

#         # Do reductions on the device where data lives
#         reduce_dims = tuple(range(mag_data.ndim - 1))

#         # These operations happen on data's device and return small tensors
#         sum_result = mag_data.sum(dim=reduce_dims)
#         sum_sq_result = (mag_data ** 2).sum(dim=reduce_dims)
#         min_result = mag_data.amin(dim=reduce_dims) if len(reduce_dims) > 0 else mag_data
#         max_result = mag_data.amax(dim=reduce_dims) if len(reduce_dims) > 0 else mag_data

#         data_count = mag_data.numel() // mag_data.shape[-1]

#         # Initialize accumulators on computation device on first update
#         if self.sum is None:
#             if self.use_gpu:
#                 # Keep on GPU for faster accumulation
#                 self.sum = torch.zeros_like(sum_result, device=self.device)
#                 self.sum_sq = torch.zeros_like(sum_sq_result, device=self.device)
#                 self.min_val = torch.full_like(min_result, float('inf'), device=self.device)
#                 self.max_val = torch.full_like(max_result, device=self.device)
#             else:
#                 # Convert to numpy for CPU
#                 num_features = len(sum_result) if hasattr(sum_result, '__len__') else 1
#                 self.sum = np.zeros(num_features)
#                 self.sum_sq = np.zeros(num_features)
#                 self.min_val = np.full(num_features, float('inf'))
#                 self.max_val = np.full(num_features, float('-inf'))

#         # Accumulate results
#         if self.use_gpu:
#             # Keep everything on GPU
#             self.sum += sum_result
#             self.sum_sq += sum_sq_result
#             self.min_val = torch.minimum(self.min_val, min_result)
#             self.max_val = torch.maximum(self.max_val, max_result)
#         else:
#             # Convert to CPU/numpy
#             sum_cpu = sum_result.detach().cpu().numpy()
#             sum_sq_cpu = sum_sq_result.detach().cpu().numpy()
#             min_cpu = min_result.detach().cpu().numpy()
#             max_cpu = max_result.detach().cpu().numpy()

#             self.sum += sum_cpu
#             self.sum_sq += sum_sq_cpu
#             self.min_val = np.minimum(self.min_val, min_cpu)
#             self.max_val = np.maximum(self.max_val, max_cpu)

#         self.count += data_count

#     def get_stats(self):
#         if self.use_gpu:
#             # Convert GPU tensors to numpy for final result
#             mean = (self.sum / self.count).cpu().numpy()
#             # Use sample variance (unbiased) with N-1 denominator
#             var = ((self.sum_sq / self.count) - (self.sum / self.count) ** 2).cpu().numpy()
#             var = var * self.count / (self.count - 1) if self.count > 1 else var
#             std = np.sqrt(np.maximum(var, 0))
#             min_val = self.min_val.cpu().numpy()
#             max_val = self.max_val.cpu().numpy()
#         else:
#             # Already on CPU as numpy
#             mean = self.sum / self.count
#             # Use sample variance (unbiased) with N-1 denominator
#             var = (self.sum_sq / self.count) - (mean ** 2)
#             var = var * self.count / (self.count - 1) if self.count > 1 else var
#             std = np.sqrt(np.maximum(var, 0))
#             min_val = self.min_val
#             max_val = self.max_val

#         # Expand magnitude statistics back to original dimensions
#         expanded_mean = self._expand_stats(mean)
#         expanded_std = self._expand_stats(std)
#         expanded_min = self._expand_stats(min_val)
#         expanded_max = self._expand_stats(max_val)

#         return {
#             'mean': expanded_mean.tolist() if hasattr(expanded_mean, 'tolist') else [expanded_mean],
#             'std': expanded_std.tolist() if hasattr(expanded_std, 'tolist') else [expanded_std],
#             'min': expanded_min.tolist() if hasattr(expanded_min, 'tolist') else [expanded_min],
#             'max': expanded_max.tolist() if hasattr(max_val, 'tolist') else [max_val]
#         }
