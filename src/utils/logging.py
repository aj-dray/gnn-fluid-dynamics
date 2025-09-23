"""Tensorboard is not currently implemented"""


# === LIBRARIES ===


import os
import json
import shutil
import sys
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path
import torch
import wandb
import numpy as np
import time
import subprocess
# from torch.utils.tensorboard import SummaryWriter

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from analysis._utils import plotting
import matplotlib.pyplot as plt
import matplotlib.colors as colors


# === CLASSES ===


class Logger:
    """Class providing features for saving model and logging progress to wandb and tensorboard"""

    def __init__(self, config, stats=None, use_wandb=True, use_tensorboard=False, resume_wandb_id=None):
        self.config = config
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.wandb_run = None
        self.tb_writer = None
        self.best_loss = np.inf
        self.norm_stats = stats
        self.resume_wandb_id = resume_wandb_id
        self.is_resuming = resume_wandb_id is not None

        # Setup logging infrastructure
        self._check_git()
        self._setup_directories()
        self._initialize_loggers()

        print(f"\nProject: {self.project}\nGroup: {self.group}\nName: {self.name}")

    @staticmethod
    def check_debug_mode_safety(config):
        """Check if debug mode is enabled in project-exec directory and exit if so."""
        current_dir = os.getcwd()
        print(f"\nCurrent directory: {current_dir}")
        root_dir = os.path.basename(current_dir)

        if root_dir == "project-exec" and config.logging.is_debug:
            print("\nERROR: Debug mode (config.logging.is_debug = True) is not allowed when running from 'project-exec' directory.")
            print("This safety check prevents debug runs in the execution environment.")
            sys.exit(1)

    def _check_git(self):
        """Get git information and save to self.git. If dirty and not debug, prompt user to continue."""
        def run_git_cmd(args):
            try:
                return subprocess.check_output(args, stderr=subprocess.STDOUT).decode().strip()
            except Exception:
                return None

        git_info = {}
        # Get commit hash
        git_info['git.commit'] = run_git_cmd(['git', 'rev-parse', 'HEAD'])
        # Get branch name
        git_info['git.branch'] = run_git_cmd(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
        # Get commit date
        git_info['git.date'] = run_git_cmd(['git', 'show', '-s', '--format=%cI', 'HEAD'])
        # Get status (dirty or clean)
        status = run_git_cmd(['git', 'status', '--porcelain'])
        git_info['git.is_dirty'] = bool(status)
        git_info['git.status'] = status

        self.git = git_info

        if git_info['git.is_dirty'] and not self.config.logging.is_debug:
            dirty_files = []
            if git_info['git.status']:
                for line in git_info['git.status'].splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        path = parts[1]
                        if path.startswith("src/"):
                            dirty_files.append(line)
            if dirty_files:
                print("\nWARNING: Git repository is dirty (uncommitted changes detected outside config/).")
                print("Branch:", git_info['git.branch'])
                print("Commit:", git_info['git.commit'])
                print("Uncommitted changes:\n", "\n".join(dirty_files))
                input("Continue anyway? [Enter to continue, Ctrl+C to abort]: ")
                print()
            else:
                self.git['git.is_dirty'] = False  # No dirty files in src/, so consider it clean

    def _setup_directories(self):
        """Setup local directory structure for organized data storage."""
        self.project = self.config.logging.project
        self.group = self.config.logging.group
        self.name = self.config.logging.name
        
        if self.is_resuming:
            # When resuming, extract timestamp from existing model path
            model_fpath = self.config.model.fpath
            if model_fpath and "(" in model_fpath and ")" in model_fpath:
                # Extract timestamp from path like ".../FvgnA(0729162152)/..."
                import re
                match = re.search(r'\((\d{10})\)', model_fpath)
                if match:
                    self.timestamp = match.group(1)
                    print(f"Extracted timestamp from model path: {self.timestamp}")
                else:
                    print("Warning: Could not extract timestamp from model path, using current time")
                    self.timestamp = datetime.now().strftime('%m%d%H%M%S')
            else:
                print("Warning: Model path format unexpected for timestamp extraction, using current time")
                self.timestamp = datetime.now().strftime('%m%d%H%M%S')
        else:
            self.timestamp = datetime.now().strftime('%m%d%H%M%S')

        if self.config.logging.is_debug:
            self.group = "debug"
            self.name = "debug"
        else:
            self.name = f"{self.name}({self.timestamp})"
        relative_dpath = f"{self.project}/{self.group}/{self.name}"
        # self.dpath_run = Path(f"_data/train/runs/{relative_dpath}")
        self.dpath_model = Path(f"../project/_data/models/{relative_dpath}")
        self.dpath_valid = Path(f"../project/_data/train/valid/{relative_dpath}")
        self.dpath_wandb = Path("../project/_data/train")
        # self.dpath_artifacts = Path("../project/_data/artifacts")

        for path in [self.dpath_model, self.dpath_wandb,
                    self.dpath_valid]:
            path.mkdir(parents=True, exist_ok=True)

    def _initialize_loggers(self):
        """Initialize TensorBoard and W&B loggers."""
        if self.use_wandb:
            wandb_config = {
                "timestamp_start": self.timestamp,
            }
            wandb_config.update(self.config.to_flat_json())
            wandb_config.update(self.git)
            
            if self.is_resuming and self.resume_wandb_id:
                # Resume existing wandb run with specific ID
                self.wandb_run = wandb.init(
                    project=self.project,
                    id=self.resume_wandb_id,
                    resume="must",
                    dir=str(self.dpath_wandb),
                    entity="ajd246-university-of-cambridge"
                )
                print(f"Resumed wandb run: {self.resume_wandb_id}")
            else:
                # Create new wandb run
                self.wandb_run = wandb.init(
                    project=self.project,
                    group=self.group,
                    name=self.name,
                    config=wandb_config,
                    dir=str(self.dpath_wandb),
                    notes=getattr(self.config.logging, 'notes', None),
                    entity="ajd246-university-of-cambridge",
                    resume="allow"
                )

    def _convert_metrics(self, metrics, prefix=""):
        flat = {}
        for k, v in metrics.items():
            key = f"{prefix}/{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(self._convert_metrics(v, prefix=key))
            elif isinstance(v, torch.Tensor):
                # If tensor is scalar, extract value; else, convert to list
                if v.numel() == 1:
                    flat[key] = v.cpu().item()
                else:
                    flat[key] = v.cpu().tolist()
            else:
                flat[key] = v
        return flat

    def save_loss(self, metrics, step, prefix):
        metrics = self._convert_metrics(metrics)
        if self.wandb_run:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
            self.wandb_run.log(metrics, step=step)

    def save_scalar(self, value, step, prefix):
        if self.wandb_run:
            metrics = {f"{prefix}": value}
            self.wandb_run.log(metrics, step=step)

    def save_plot(self, array_x, array_y, labels, step, prefix):
        xlabel, ylabel = labels
        data = [[x, y] for x, y in zip(array_x, array_y)]
        table = wandb.Table(data=data, columns=[xlabel, ylabel])
        plot = wandb.plot.line(table, x=xlabel, y=ylabel, title=prefix)
        self.wandb_run.log({f"{prefix}": plot}, step=step)

    def save_plots(self, evolution_arrays, step, prefix="valid"):
        """Save evolution arrays as line plots to wandb."""
        if not self.wandb_run:
            return

        for metric_name, evolutions in evolution_arrays.items():
            # Plot overall evolution (mean across all simulations)
            if "evo_all" in evolutions:
                timesteps = list(range(1, len(evolutions["evo_all"])))
                self.save_plot(timesteps, evolutions["evo_all"],
                             labels=("Timestep", f"{metric_name.replace('_', ' ').title()}"),
                             step=step, prefix=f"{prefix}/{metric_name}/evolution_mean")

            # Plot individual simulation evolutions
            for sim_key, sim_evolution in evolutions.items():
                if sim_key.startswith("evo_") and sim_key != "evo_all":
                    sim_id = sim_key[4:]  # Remove "evo_" prefix
                    self.save_plot(timesteps, sim_evolution,
                                 labels=("Timestep", f"{metric_name.replace('_', ' ').title()}"),
                                 step=step, prefix=f"{prefix}/{metric_name}/evolution_{sim_id}")

    def save_snapshot(self, snapshot_data, step, prefix="rollout"):
        """Save snapshot plots to wandb."""
        if not self.wandb_run or not snapshot_data:
            return
        # Iterate over timesteps
        for timestep, mesh_data in snapshot_data.items():
            # Iterate over meshes in this timestep
            for mesh_name, data in mesh_data.items():
                # Create the geometry dict expected by plot_cell_field
                geom = {
                    'vertex_pos': data['vertex_pos'],
                    'vertex_face': data['vertex_face']
                }

                # Create figure
                fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=100)

                # Plot the velocity field
                field = data['field_data']
                velocity_magnitude = np.linalg.norm(field, axis=1)

                # Create colormap and normalization
                vmin, vmax = velocity_magnitude.min(), velocity_magnitude.max()
                norm = colors.Normalize(vmin=vmin, vmax=vmax)
                cmap = plt.cm.viridis

                # Plot
                plotting.plot_cell_field(ax, field, geom, cmap, norm,
                                label=f'Cell Velocity Field t={timestep} {mesh_name}',
                                overlay_mesh=True, show_label=True)

                # Save to wandb with mesh name appended
                plot_name = f"{prefix}/cell_velocity_field_t{timestep}_{mesh_name}"
                # print(f"saving {plot_name}")
                self.wandb_run.log({plot_name: wandb.Image(fig)}, step=step)

                plt.close(fig)

    def save_model(self, config, model, optimizer, scheduler, epoch, mini_epoch, step,
                   train_losses, valid_losses):
        """Save model checkpoint with wandb integration and local backup."""
        save_time = time.time()
        save_overwrite = config.logging.save_overwrite

        checkpoint = {
            'epoch': epoch,
            'mini_epoch': mini_epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else self.config.__dict__,
            'timestamp': self.timestamp,
            'wandb_run_id': self.wandb_run.id if self.wandb_run else None,
            'stats': self.norm_stats
        }

        if save_overwrite:
            # Always save the latest model
            latest_fpath = os.path.join(self.dpath_model, f"checkpoint-{mini_epoch}(latest).pth")
            torch.save(checkpoint, latest_fpath)

            # Check if this is the best model
            current_loss = valid_losses["total_mean_error"]
            is_best = self.best_loss > current_loss

            if is_best:
                self.best_loss = current_loss
                for file in os.listdir(self.dpath_model):
                    if file.startswith("checkpoint-") and file.endswith("(best).pth"):
                        os.remove(os.path.join(self.dpath_model, file))
                best_fpath = os.path.join(self.dpath_model, f"checkpoint-{mini_epoch}(best).pth")
                torch.save(checkpoint, best_fpath)
                print(f"\t{'save':<5} | {'e':<1} {epoch:>3} | {'me':<2} {mini_epoch:>5} | {'s':<1} {step:>6} | {'t':<1} {(time.time() - save_time):<3.2e} | best")

                # Upload best model to wandb
                if self.wandb_run:
                    artifact = wandb.Artifact(
                        name=f"checkpoint_{mini_epoch}",
                        type="model"
                    )
                    artifact.add_file(str(best_fpath))
                    self.wandb_run.log_artifact(artifact)
            else:
                print(f"\t{'save':<5} | {'e':<1} {epoch:>3} | {'me':<2} {mini_epoch:>5} | {'s':<1} {step:>6} | {'t':<1} {(time.time() - save_time):<3.2e} | latest")

            # Clean up old checkpoint files (keep only current latest and best)
            for file in os.listdir(self.dpath_model):
                if file.startswith("checkpoint-") and file.endswith(".pth"):
                    if (file != f"checkpoint-{mini_epoch}(latest).pth" and
                        not file.endswith("(best).pth")):
                        os.remove(os.path.join(self.dpath_model, file))
        else:
            # Save with mini_epoch naming (keeps everything)
            save_fpath = os.path.join(self.dpath_model, f"checkpoint-{mini_epoch}.pth")
            torch.save(checkpoint, save_fpath)
            print(f"\t{'save':<5} | {'e':<1} {epoch:>3} | {'me':<2} {mini_epoch:>5} | {'s':<1} {step:>6} | {'t':<1} {(time.time() - save_time):<3.2e}")

            if self.wandb_run:
                artifact = wandb.Artifact(
                    name=f"checkpoint_{mini_epoch}",
                    type="model"
                )
                artifact.add_file(str(save_fpath))
                self.wandb_run.log_artifact(artifact)

    def set_norm_stats(self, stats):
        self.norm_stats = stats
        if self.wandb_run:
            name = self.name.split("(")[0]  # Use name without timestamp")
            artifact = wandb.Artifact(
                name=f"stats_{name}",
                type="stats"
            )
            # Save stats to the specified file path
            stats_path = self.config.dataset.stats_fpath
            artifact.add_file(stats_path)
            self.wandb_run.log_artifact(artifact)

    def watch_model(self, model, frequency):
        self.wandb_run.watch(model, log="all", log_freq=frequency)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up"""
        if self.wandb_run is not None:
            self.wandb_run.finish()
        if self.tb_writer is not None:
            self.tb_writer.close()
