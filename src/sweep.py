#!/usr/bin/env python3
"""
Hyperparameter sweep script for training jobs.
Distributes sweep values across SLURM array jobs.
"""

import json
import os
import sys
import argparse
import subprocess


def set_nested_value(config_dict, key_path, value):
    """
    Set a nested value in a dictionary using dot notation.
    
    Args:
        config_dict: Dictionary to modify
        key_path: Dot-separated path (e.g., 'model.mp_num')
        value: Value to set
    """
    keys = key_path.split('.')
    current = config_dict

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value


def run_training_job(base_config_path, sweep_config, param_combination, run_id):
    """
    Execute a training run with modified hyperparameters.
    
    Args:
        base_config_path: Path to base configuration file
        sweep_config: Sweep configuration dictionary
        param_combination: Parameter values for this run
        run_id: Unique identifier for this run
        
    Returns:
        Return code from training process
    """

    # Load base config
    with open(base_config_path, 'r') as f:
        config = json.load(f)

    # Handle both single parameter (backward compatibility) and multiple parameters
    if 'sweep_parameter' in sweep_config:
        # Single parameter mode (backward compatibility)
        set_nested_value(config, sweep_config['sweep_parameter'], param_combination)
        param_name = sweep_config['sweep_parameter'].split('.')[-1]
        original_name = config['logging']['name']
        config['logging']['name'] = f"{param_name}:{param_combination}-{original_name}"
    else:
        # Multiple parameters mode
        name_parts = []
        for param_name, param_value in param_combination.items():
            set_nested_value(config, param_name, param_value)
            short_name = param_name.split('.')[-1]  # Get last part of dot notation
            name_parts.append(f"{short_name}:{param_value}")
        
        original_name = config['logging']['name']
        param_prefix = "-".join(name_parts)
        config['logging']['name'] = f"{param_prefix}-{original_name}"

    # Create temporary config file
    temp_config_path = f"../project/config/tmp/sweep_config_{os.getpid()}_{run_id}.json"
    with open(temp_config_path, 'w') as f:
        json.dump(config, f, indent=2)

    try:
        # Run training
        cmd = [sys.executable, "src/train.py", "--config", temp_config_path]
        print(f"Running: {' '.join(cmd)}")
        
        if 'sweep_parameter' in sweep_config:
            print(f"Sweep parameter {sweep_config['sweep_parameter']} = {param_combination}")
        else:
            print(f"Sweep parameters: {param_combination}")

        result = subprocess.run(cmd, check=True)
        return result.returncode

    finally:
        # Cleanup temp file
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


def generate_parameter_combinations(sweep_config):
    """Generate all parameter combinations based on sweep configuration."""
    
    # Single parameter mode (backward compatibility)
    if 'sweep_parameter' in sweep_config:
        return sweep_config['sweep_values']
    
    # Multiple parameters mode
    elif 'parameters' in sweep_config:
        parameters = sweep_config['parameters']
        
        # Check if we have explicit combinations
        if 'combinations' in sweep_config:
            return sweep_config['combinations']
        
        # Generate cartesian product of all parameter values
        import itertools
        
        param_names = list(parameters.keys())
        param_values = [parameters[param] for param in param_names]
        
        combinations = []
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)
        
        return combinations
    
    else:
        raise ValueError("Sweep config must contain either 'sweep_parameter' (single param) or 'parameters' (multi param)")


def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter sweep')
    parser.add_argument('sweep_config', type=str, help='Path to sweep configuration file')
    parser.add_argument('--array_id', type=int, default=0,
                       help='Array job ID (0-based)')
    parser.add_argument('--array_total', type=int, default=1,
                       help='Total number of array jobs')
    args = parser.parse_args()

    # Load sweep configuration
    with open(args.sweep_config, 'r') as f:
        sweep_config = json.load(f)

    sweep_combinations = generate_parameter_combinations(sweep_config)
    total_sweeps = len(sweep_combinations)

    # Distribute sweep values across array jobs
    sweeps_per_job = total_sweeps // args.array_total
    remainder = total_sweeps % args.array_total

    # Calculate which sweep values this job should handle
    start_idx = args.array_id * sweeps_per_job + min(args.array_id, remainder)
    if args.array_id < remainder:
        end_idx = start_idx + sweeps_per_job + 1
    else:
        end_idx = start_idx + sweeps_per_job

    my_sweep_combinations = sweep_combinations[start_idx:end_idx]

    print(f"Array job {args.array_id}/{args.array_total}")
    print(f"Running sweeps {start_idx}-{end_idx-1} ({len(my_sweep_combinations)} total)")

    # Run assigned sweep combinations
    for i, param_combination in enumerate(my_sweep_combinations):
        global_idx = start_idx + i
        print(f"\n=== Running sweep {global_idx+1}/{total_sweeps} (local {i+1}/{len(my_sweep_combinations)}) ===")
        try:
            run_training_job(
                sweep_config['base_config'],
                sweep_config,
                param_combination,
                global_idx
            )
        except subprocess.CalledProcessError as e:
            print(f"Error in sweep {global_idx+1}: {e}")
            sys.exit(1)

    print(f"\nCompleted {len(my_sweep_combinations)} sweep runs for array job {args.array_id}")


if __name__ == "__main__":
    main()
