"""
Script for training.
Handles resuming training runs.
Distributed training not operational since update to rollout validation.
"""


# === LIBRARIES ===


import os
import sys
import json
import time
from datetime import datetime
from typing import Optional
from importlib import import_module
import shutil
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import RandomSampler, SequentialSampler, BatchSampler
import argparse
import traceback
import numpy as np
import pprint
# torch.autograd.set_detect_anomaly(True)


# === MODULES ===


from utils.config import Config
import utils.lr_schedule as schedulers
from utils.logging import Logger
from utils.sampler import RolloutSampler
from utils.monitoring import ModelMonitor
from rollout import Rollout
from utils.loss import MSE_per_element, MSE_per_element_torch, MSE_per_batch_torch
from utils.model_loading import merge_checkpoint_config, load_model_state_dict_flexible


# === FUNCTIONS ===


def multi_mean(sum_losses, num_samples):
    stats = torch.empty(sum_losses.numel() + 1,
                        device=sum_losses.device,
                        dtype=sum_losses.dtype)
    stats[:-1] = sum_losses
    stats[-1]   = num_samples

    dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    global_sum   = stats[:-1]
    global_count = stats[-1]
    return global_sum / global_count


def multi_process_mean(tensor):
    """Average a tensor across all ranks."""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()  # Add this line to get average
    return rt


def select_optimizer(config, model):
    """
    Create optimizer based on configuration.

    Args:
        config: Configuration object with optimizer settings
        model: PyTorch model to optimize

    Returns:
        PyTorch optimizer instance
    """
    training = config.training
    if training.optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training.lr_max,
            weight_decay=training.weight_decay if training.weight_decay is not None else 0
        )
    elif training.optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training.lr_max
        )
    else:
        raise ValueError(f"Optimizer {training.optimizer_name} not recognised")
    return optimizer


def clear_gpu_memory():
    torch.cuda.empty_cache()


def get_gpu_memory_info():
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"GPU memory allocated: {allocated:.2f} MB")
    print(f"GPU memory reserved: {reserved:.2f} MB")


# === CLASSES ===


class Trainer():
    """
    Manages the training loop, validation, and logging for neural network models.
    Handles distributed training setup and model monitoring.
    """
    def __init__(self, config, device, rank, optimizer, scheduler, validator, stats, resume_wandb_id=None):
        """
        Initialize trainer with training components.

        Args:
            config: Configuration object with training parameters
            device: PyTorch device for training
            rank: Process rank for distributed training
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler
            validator: Rollout instance for validation
            stats: Dataset statistics for normalization
            resume_wandb_id: Optional wandb run ID for resuming
        """
        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.validator = validator
        self.rank = rank
        self.device = device
        self.is_master = (rank == 0)
        self.use_ddp = self.config.settings.multi_gpu
        self.world_size = self.config.settings.num_gpus if self.use_ddp else 1

        # Initialize training counters
        self.mini_epoch_count = 0
        self.epoch_count = 0
        self.step_count = 0
        self.sample_count = 0

        # Initialize monitoring
        self.monitor = ModelMonitor()

        self.config.logging.run_count += 1 # for name
        if self.is_master and not self.config.logging.is_debug:
            self.logger = Logger(self.config, use_wandb=self.config.logging.use_wandb, use_tensorboard=self.config.logging.use_tensorboard, resume_wandb_id=resume_wandb_id)
            self.logger.set_norm_stats(stats)
        else:
            self.logger = None

        print(f"Trainer setup on {self.device}.\n")

    def run(self, model, train_loader, valid_loader):
        """Runs full training loop"""
        # Setup
        self.run_start = time.time()
        if self.logger:
            self.logger.watch_model(model, frequency=self.config.logging.save_frequency)
        model = model.module if self.use_ddp else model
        mini_epoch_step_count = self.config.training.mini_epoch_size / self.config.training.batch_size

        # Pre-training valid errors
        valid_losses = self._validate(model, valid_loader)
        if self.logger:
            self.logger.save_loss(valid_losses, step=self.mini_epoch_count, prefix="valid")

        # Training loop
        mini_epoch_losses = {}
        valid_losses = {}  # Initialize to avoid scope issues
        mini_epoch_train_start = time.time()
        if self.is_master:
            print("\nTraining start...\n")
        for _ in range(self.config.training.epochs - self.epoch_count):
            self.epoch_count += 1
            if self.use_ddp:
                train_loader.sampler.set_epoch(self.epoch_count)
            for i, data_batch in enumerate(train_loader):
                batch = [graph.to(self.device) for graph in data_batch] # creates input features for model specifically.
                self.step_count += 1
                self.sample_count += len(batch[0])
                train_losses = self._train_step(model, batch)

                # Reduce losses across all processes for distributed training
                if self.use_ddp:
                    for k, v in train_losses.items():
                        reduced_loss = multi_process_mean(v)
                        mini_epoch_losses[k] = mini_epoch_losses.get(k, 0.0) + reduced_loss.item()
                else:
                    for k, v in train_losses.items():
                        mini_epoch_losses[k] = mini_epoch_losses.get(k, 0.0) + v.item()

                # Mini-Epoch
                if self.step_count % (self.config.training.mini_epoch_size // self.config.training.batch_size) == 0:
                    self.mini_epoch_count += 1

                    # Calculate training time
                    mini_epoch_train_end = time.time()
                    mini_epoch_train_time = mini_epoch_train_end - mini_epoch_train_start
                    train_step_time = mini_epoch_train_time / mini_epoch_step_count

                    # Train losses
                    for k in mini_epoch_losses:
                        mini_epoch_losses[k] /= mini_epoch_step_count

                    if self.logger:
                        self.logger.save_loss(mini_epoch_losses, step=self.mini_epoch_count, prefix="train")
                        self.logger.save_scalar(train_step_time, step=self.mini_epoch_count, prefix="performance/train_step_time")
                        self.logger.save_scalar(mini_epoch_train_time, step=self.mini_epoch_count, prefix="performance/mini_epoch_train_time")
                    if self.is_master:
                        total_log_loss = mini_epoch_losses['total_log_loss']
                        print(f"\t{'train':<5} | {'e':<1} {self.epoch_count:>3} | {'me':<2} {self.mini_epoch_count:>5} | {'s':<1} {self.step_count:>6} | {'t':<1} {mini_epoch_train_time:<3.2e} | {'loss':<4} {total_log_loss:>3.2e} | {'lr':<2} {self.optimizer.param_groups[0]['lr']:>3.2e}")
                        if self.config.logging.is_debug:
                            formatted_output = pprint.pformat(mini_epoch_losses, indent=2)
                            indented_output = '\n'.join('\t\t' + line for line in formatted_output.split('\n'))
                            print(indented_output)

                    # Validation
                    if self.mini_epoch_count % self.config.logging.valid_frequency == 0:
                        valid_losses = self._validate(model, valid_loader)
                        if self.logger:
                            self.logger.save_loss(valid_losses, step=self.mini_epoch_count, prefix="valid")

                    # Save model
                    if self.config.logging.save_frequency != 0 and self.mini_epoch_count % self.config.logging.save_frequency == 0 and self.logger:
                        self.logger.save_model(self.config, model, self.optimizer, self.scheduler, self.epoch_count, self.mini_epoch_count, self.step_count, mini_epoch_losses, valid_losses)

                    # Updates
                    self.scheduler.step()
                    if self.logger:
                        self.logger.save_scalar(self.optimizer.param_groups[0]['lr'], step=self.mini_epoch_count, prefix="train/learning_rate")
                        self.logger.save_scalar(self.sample_count, step=self.mini_epoch_count, prefix="train/sample_count")

                    # Reset for next mini-epoch
                    mini_epoch_train_start = time.time()
                    mini_epoch_losses = {}

        print(f"\nTraining complete | time = {(time.time() - self.run_start):.3e} seconds.\n")

    def _train_step(self, model, batch):
        model.train()
        if self.config.training.pushforward_factor and model.pushforward_use:
            for _ in range(self.config.training.pushforward_factor):
                batch = self._rollout_step(model, batch) # no gradient
            batch[0].y[:, 0:2] = batch[0].y[:, 0:2] - batch[0].x[:, 0:2] # set target
            output = model(batch, mode='train')
        else:
            output = model(batch, mode='train')
        self.optimizer.zero_grad()
        train_losses = model.loss(output, batch)
        train_losses["total_log_loss"].backward()

        # Monitor decoder.face_mlp gradients and scalar parameters
        actual_model = model.module if self.use_ddp else model
        if self.logger and hasattr(actual_model, 'decoder') and hasattr(actual_model.decoder, 'face_mlp'):
            self.monitor.monitor_face_mlp_gradients(actual_model.decoder.face_mlp, self.logger, self.mini_epoch_count)

        # Monitor scalar parameters if they exist
        if self.logger:
            self.monitor.monitor_scalar_parameters(actual_model, self.logger, self.mini_epoch_count)

        # Clip gradients
        if self.config.training.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.training.clip_grad_norm)

        # Update gradients
        self.optimizer.step()

        # Monitor decoder.face_mlp updates after optimizer step
        actual_model = model.module if self.use_ddp else model
        if self.logger and hasattr(actual_model, 'decoder') and hasattr(actual_model.decoder, 'face_mlp'):
            self.monitor.monitor_face_mlp_updates(actual_model.decoder.face_mlp, self.logger, self.mini_epoch_count)

        # Update logging
        if self.is_master and self.config.logging.loss_frequency and self.mini_epoch_step_count % self.config.logging.loss_frequency == 0:
            total_log_loss = train_losses.get('total_log_loss')
            print(f"\t{' ':<5} | {'e':<1} {self.epoch_count:>3} | {'me':<2} {self.mini_epoch_count:>5} | {'s':<1} {self.step_count:>6} | {'t':<1} {(time.time() - self.run_start):<3.2e} | {'loss':<4} {total_log_loss:>3.2e}") # log scale loss

        return train_losses

    def _validate(self, model, valid_loader):
        if self.validator:
            valid_start_time = time.time()
            if self.logger:
                run_name = self.logger.name
            else:
                run_name = "debug"
            save_status = 'snapshot' if len(self.config.rollout.snapshot_indices) > 0 else 'off'
            scalar_losses, evolution_arrays, snapshot_data = self.validator.run(model, valid_loader, save=save_status, error="on", output=run_name)
            error = scalar_losses["total_mean_error"]
            valid_runtime = time.time() - valid_start_time
            print(f"\t{'valid':<5} | {'e':<1} {self.epoch_count:>3} | {'me':<2} {self.mini_epoch_count:>5} | {'s':<1} {self.step_count:>6} | {'t':<1} {valid_runtime:<3.2e} | {'error':<5} {error:>3.2e}") # ignore divergence
            if self.logger:
                self.logger.save_scalar(valid_runtime, step=self.mini_epoch_count, prefix="performance/valid_time")
                self.logger.save_plots(evolution_arrays, step=self.mini_epoch_count, prefix="rollout") # evolution of RelMSE error for velocity, pressure, divergence
                self.logger.save_snapshot(snapshot_data, step=self.mini_epoch_count, prefix="rollout") # image of velocity field
            return scalar_losses
        return {}

    def _rollout_step(self, model, input_graphs):
        with torch.no_grad():
            output = model([graph.clone() for graph in input_graphs], mode='rollout')
            solutions = output
            if "cell_velocity" in output:
                solutions["cell_velocity"] = output["cell_velocity"]
            elif "cell_velocity_change" in output:
                solutions["cell_velocity"] = input_graphs[0].x[:, 0:2] + output["cell_velocity_change"]
            return model.update_features(solutions, input_graphs)

# === MAIN ===


def main():

    # CL arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()

    # Config
    with open(args.config, 'r') as f:
        data = json.load(f)
    config = Config.from_dict(data)

    # Load checkpoint early and merge configs if resuming
    checkpoint = None
    if config.model.fpath is not None:
        print(f"Loading checkpoint from: {config.model.fpath}")
        checkpoint = torch.load(config.model.fpath, map_location='cpu', weights_only=False)
        config = merge_checkpoint_config(config, checkpoint)

    Logger.check_debug_mode_safety(config) # prevent debug mode in execution environment

    # Random
    seed = config.settings.random_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Import modules and classes
    model_module = import_module(config.model.module)
    model_class = getattr(model_module, config.model.name)
    transform_fn = model_class.transform_features # for creating input features
    dataset_module = import_module(config.dataset.module)
    dataset_class = getattr(dataset_module, config.dataset.name)

    # Adjust config given model
    config.training.pushforward_factor = config.training.pushforward_factor if model_class.pushforward_use else None

    # Train Dataset
    data_directory = config.dataset.dpath
    shuffle = config.dataset.shuffle
    train_dataset = dataset_class(data_directory, config, mode='train', noise=True, shuffle=shuffle, transform_fn=transform_fn)
    print("\tTrain meshes: ", train_dataset.get_sim_ids())

    # Stats Accumulation
    stats_fpath = config.dataset.stats_fpath
    config.preproc.data_subset = config.training.data_subset # for preproc mode
    config.preproc.data_timestep_range = config.training.data_timestep_range
    config.preproc.data_sim_limit = config.training.data_sim_limit
    tmp_dataset = dataset_class(data_directory, config, mode='stats', noise=False, shuffle=False, transform_fn=transform_fn)
    stats = tmp_dataset.read_stats(stats_fpath, model_class)

    # Model
    model = model_class(config, MSE_per_element_torch, train_dataset, stats)

    # Apply checkpoint state if loaded
    start_epoch = 0
    start_mini_epoch = 0
    start_step = 0
    resume_wandb_id = None
    if checkpoint is not None:
        load_model_state_dict_flexible(model, checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_mini_epoch = checkpoint['mini_epoch']
        start_step = checkpoint['step']
        resume_wandb_id = checkpoint.get('wandb_run_id', None)
        print(f"Resuming from epoch {start_epoch}, mini-epoch {start_mini_epoch}, step {start_step}")
        if resume_wandb_id:
            print(f"Will resume wandb run: {resume_wandb_id}")

    # Update Dataset
    print("Setting dataset noise std and grad weights...")
    train_dataset.set_noise_std(stats)
    train_dataset.set_grad_weights(model)

    # Settings
    num_workers = config.training.num_workers
    pin_memory = config.settings.pin_memory
    device = config.settings.device
    use_ddp = config.settings.multi_gpu and device

    # Multi-GPU Dataloader
    if use_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rank = dist.get_rank()
        world_size = config.settings.num_gpus
        assert world_size == torch.cuda.device_count()
        batch_size = config.training.batch_size // world_size # distribute between

        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        model = model.to(device)
        model = DDP(model, device_ids=[local_rank])

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=config.training.persistent_workers)

    # Single-GPU Dataloader
    else:
        rank = 0
        model = model.to(device)
        train_sampler = BatchSampler(RandomSampler(train_dataset), batch_size=config.training.batch_size, drop_last=False)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, pin_memory=pin_memory, num_workers=num_workers, persistent_workers=config.training.persistent_workers, prefetch_factor=config.training.prefetch_factor)

    # Optimizer
    optimizer = select_optimizer(config, model)
    scheduler_class = getattr(schedulers, config.training.lr_class)
    samples_per_epoch = len(train_dataset)
    mini_epoch_size = config.training.mini_epoch_size
    total_samples = samples_per_epoch * config.training.epochs
    total_mini_epochs = total_samples // mini_epoch_size
    lr_scheduler = scheduler_class(optimizer, config.training, total_mini_epochs)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] is not None:
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Single-GPU Validation (always on rank 0)
    num_workers = config.rollout.num_workers
    if rank == 0:
        # Dataset
        valid_dataset = dataset_class(data_directory, config, mode='rollout', noise=False, shuffle=False, transform=None)
        valid_dataset.set_grad_weights(model)
        sim_ids = valid_dataset.get_sim_ids()
        print("\tValid meshes: ", sim_ids)

        # Dataloader
        valid_sampler = BatchSampler(RolloutSampler(valid_dataset, shuffle=False), batch_size=config.rollout.batch_size, drop_last=False)
        valid_loader = DataLoader(valid_dataset, batch_sampler=valid_sampler, pin_memory=pin_memory, num_workers=num_workers, persistent_workers=config.rollout.persistent_workers, prefetch_factor=config.rollout.prefetch_factor)

        # Rollout
        rollout_device = config.settings.device
        validator = Rollout(config, rollout_device, sim_ids)

    else:
        validator = None

    # Trainer
    print()
    trainer = Trainer(config, device, rank, optimizer, lr_scheduler, validator, stats, resume_wandb_id)
    if checkpoint is not None: # set states if resuming
        trainer.epoch_count = start_epoch
        trainer.mini_epoch_count = start_mini_epoch
        trainer.step_count = start_step
        trainer.sample_count = start_step * config.training.batch_size

    trainer.run(model, train_loader, valid_loader)

    # Clean up
    if use_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining stopped by keyboard interrupt.")
        sys.exit(1)
    except Exception as e:
        print("\nTraining failed:")
        traceback.print_exc()
        sys.exit(1)
