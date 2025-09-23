"""
Script for rolling out simulations.
Class also used in train.py for validation
"""


# === LIBRARIES ===


import torch
import os
import argparse
import importlib
import numpy as np
import h5py
import time
import json
import sys
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch.utils.data import BatchSampler
import traceback


# === MODULES ===


from utils.simulation_data import SimulationData
from utils.config import Config
from utils.geometry import interpolate_face_to_centroid
import utils.loss
from utils.loss import MSE_per_element
import utils.lr_schedule as schedulers
from utils.sampler import RolloutSampler
import utils.maths
from utils.model_loading import initialise_model, update_config
import utils.fvm as fvm


# === CLASSES ===


class Rollout:
    """
    Handles model rollout simulations for validation and evaluation.
    Performs autoregressive timestep predictions and computes error metrics.
    """
    def __init__(self, config, device, sim_ids):
        """
        Initialize rollout handler.
        
        Args:
            config: Configuration object with rollout parameters
            device: PyTorch device for computations
            sim_ids: List of simulation IDs to process
        """
        self.config = config
        self.device = device
        self.data_timestep_range = config.rollout.data_timestep_range
        self.batch_size = config.rollout.batch_size
        self.sim_ids = sim_ids

        assert len(self.sim_ids) == config.rollout.batch_size, "sim_ids length must equal batch_size"

    def _simulation_setup(self, save, output, initial_data, num_timesteps, batches):
        """Setup output files and metadata for rollout simulation."""
        dpath = "../project/_data/rollout"
        parts = os.path.normpath(self.config.model.fpath).split(os.sep)
        project_model, group_model, name_model = parts[2:5]
        project = self.config.logging.project if self.config.logging.project else project_model
        group = self.config.logging.group if self.config.logging.group else group_model
        name = self.config.logging.name if self.config.logging.name else f"{name_model}-{self.config.rollout.data_subset}-{self.config.logging.notes}"
        output_dir = os.path.join(dpath, project, group, name)
        os.makedirs(output_dir, exist_ok=True)

        data_fpath = os.path.join(output_dir, 'data0.h5')
        data_file = h5py.File(data_fpath, 'w')
        meta_file = os.path.join(output_dir, 'meta.json')
        print("\tData file: ", data_fpath)
        meta_data = {
            "model": self.config.model.fpath,
            "dataset": self.config.dataset.dpath,
            "subset": self.config.rollout.data_subset,
            "timerange": self.config.rollout.data_timestep_range,
            "save_type": save,
            "meshes": {
                "data0":  [sim_id.tolist() if isinstance(sim_id, np.ndarray) else sim_id for sim_id in self.sim_ids]
            },
            "notes": self.config.logging.notes if self.config.logging.notes else "",
        }

        if save == "full":
            # Initial conditions + timesteps saved at save_frequency intervals
            num_saved_timesteps = 1 + num_timesteps // self.config.rollout.save_frequency
            simulation_data = SimulationData(data_file, initial_data, self.sim_ids, num_saved_timesteps)
        else:
            num_saved_timesteps = 1


        # Initial conditions
        if save == "full":
            initial_solutions = {
                "cell_velocity": initial_data[0].velocity[:, 0],
                "cell_pressure": initial_data[0].pressure[:, 0],
                "face_velocity": initial_data[1].velocity[:, 0],
                "face_pressure": initial_data[1].pressure[:, 0],
                "face_flux": torch.zeros(initial_data[1].pos.shape[0], 1)  # Initial flux is zero
            }
            # For initial conditions, save ground truth data as both prediction and ground truth
            simulation_data.save_timestep(initial_solutions, batches, self.sim_ids, 0, self.data_timestep_range[0], ground_truth=initial_solutions)

        return simulation_data, data_file, meta_data, meta_file

    def _simulation_cleanup(self, simulation_file, meta_file, meta_data, start_time):
        meta_data["run_time"] = time.time() - start_time
        with open(meta_file, 'w') as f:
            json.dump(meta_data, f, indent=2)
        simulation_file.close()

    def _error_accumulate(self, batch_idx, data, solutions, input_graphs, batches, loss_cell_velocity, loss_cell_pressure, loss_divergence, target_index=-1):
        """Compute and accumulate velocity, pressure, and divergence errors."""

        target_cell_velocity = data[0].velocity[:, target_index]
        # print(target_cell_velocity.shape)
        # print(solutions["cell_velocity"].shape)
        # print(data[0].velocity.shape)
        loss_cell_velocity[batch_idx] = utils.loss.RelMSE_per_graph(target_cell_velocity, solutions["cell_velocity"], batch=batches[0])

        target_cell_pressure = data[0].pressure[:, target_index]
        loss_cell_pressure[batch_idx] = utils.loss.RelMSE_per_graph(target_cell_pressure, solutions["cell_pressure"], batch=batches[0])

        if "cell_flux" in solutions:
            divergence_error = fvm.divergence_from_cell_flux(solutions["cell_flux"])
        # elif "face_flux" in solutions:
        #     divergence_error = fvm.divergence_from_face_flux(solutions["face_flux"], input_graphs[1].face)
        elif "face_velocity" in solutions:
            boundary_mask = input_graphs[1].boundary_mask # apply boundary conditions
            if len(input_graphs[1].y.shape) > 2:
                solutions["face_velocity"][boundary_mask] = input_graphs[1].y[boundary_mask, target_index, 0:2]
            else:
                solutions["face_velocity"][boundary_mask] = input_graphs[1].y[boundary_mask, 0:2]
            divergence_error = fvm.divergence_from_uf(solutions["face_velocity"], input_graphs[0].normal, input_graphs[1].area, input_graphs[1].face)
        elif "cell_velocity" in solutions:
            divergence_error = fvm.divergence_from_uc(solutions["cell_velocity"], input_graphs[0].grad_weights, input_graphs[0].grad_neighbours, input_graphs[0].volume)
        else:
            divergence_error = torch.zeros_like(input_graphs[0].volume)
        loss_divergence[batch_idx] = utils.loss.MSE_per_graph(divergence_error, torch.zeros_like(divergence_error), batch=batches[0])

    def _simulations_save(self, simulation_data, solutions, batches, sim_ids, save, i, save_index, data_timestep_range, data, config):
        # Extract ground truth data from the loaded data
        ground_truth = {
            "cell_velocity": data[0].velocity[:, -1],
            "cell_pressure": data[0].pressure[:, -1],
            "face_velocity": data[1].velocity[:, -1],
            "face_pressure": data[1].pressure[:, -1],
            # "dt": data[0].dt if hasattr(data[0], 'dt') else None
        }
        if (save == "full" and i % config.rollout.save_frequency == 0):
            simulation_data.save_timestep(
                solutions, batches, sim_ids, index=save_index,
                timestep=i+1+data_timestep_range[0], ground_truth=ground_truth
            )
            return save_index + 1
        return save_index

    def _error_save(self, loss_cell_velocity, loss_cell_pressure, loss_divergence, dataloader):
        sim_ids = dataloader.dataset.get_sim_ids()

        def compute_scalar_metrics(loss_tensor):
            # Overall statistics across all timesteps and batches
            mean_all = torch.mean(loss_tensor).cpu().item()
            max_all = torch.max(loss_tensor).cpu().item()

            # Variance of evolution across simulations (how much simulations differ in their error patterns)
            # This measures consistency between different simulations over time
            sim_means = torch.mean(loss_tensor, dim=0)  # Mean error per simulation across time
            variance_mean_all = torch.var(sim_means).cpu().item()  # variance of mean across timesteps
            sim_vars = torch.var(loss_tensor, dim=1)  # variance across simulations at each timestep
            mean_variance_all = torch.mean(sim_vars).cpu().item()  # mean of variance at each timestep

            return {
                "mean_all": mean_all,
                "max_all": max_all,
                "mean_variance_all": mean_variance_all,
                "variance_mean_all": variance_mean_all
            }

        def compute_evolution_arrays(loss_tensor):
            # Evolution across time for each batch (per simulation)
            evo_per_sim = {f"evo_{sim_ids[i]}": loss_tensor[:, i].cpu().tolist()
                          for i in range(loss_tensor.shape[1])}

            # Evolution of mean across all batches at each timestep
            evo_all = torch.mean(loss_tensor, dim=1).cpu().tolist()

            return {
                "evo_all": evo_all,
                **evo_per_sim
            }

        # Separate scalar metrics and evolution arrays
        scalar_losses = {
            "velocity_error": compute_scalar_metrics(loss_cell_velocity),
            "pressure_error": compute_scalar_metrics(loss_cell_pressure),
            "divergence_error": compute_scalar_metrics(loss_divergence),
            "total_mean_error": torch.mean(loss_cell_velocity + loss_cell_pressure).cpu().item()
        }

        evolution_arrays = {
            "velocity_error": compute_evolution_arrays(loss_cell_velocity),
            "pressure_error": compute_evolution_arrays(loss_cell_pressure),
            "divergence_error": compute_evolution_arrays(loss_divergence)
        }

        if self.config.logging.is_debug:
            print(f"\t\tvelocity_error (mean_all): {scalar_losses['velocity_error']['mean_all']}")
            print(f"\t\tpressure_error (mean_all): {scalar_losses['pressure_error']['mean_all']}")
            print(f"\t\tdivergence_error (mean_all): {scalar_losses['divergence_error']['mean_all']}")
            # print(f"\t\tvelocity variance across sims: {scalar_losses['velocity_error']['variance_all']}")
            # print(f"\t\tpressure variance across sims: {scalar_losses['pressure_error']['variance_all']}")

        return scalar_losses, evolution_arrays

    def _save_snapshot(self, solutions, input_graphs, timestep, dataloader):
        """Save snapshot data for plotting."""
        # Get mesh names from dataset
        sim_ids = dataloader.dataset.get_sim_ids()

        # Unbatch the graphs to get individual samples
        # c_graph_list = Batch.to_data_list(input_graphs[0])
        # f_graph_list = Batch.to_data_list(input_graphs[1])
        v_graph_list = Batch.to_data_list(input_graphs[2])

        # Unbatch solutions
        cell_batch = input_graphs[0].batch.cpu().numpy()
        cell_velocity_list = []
        for batch_idx in range(self.batch_size):
            cell_mask = cell_batch == batch_idx
            cell_velocity_list.append(solutions["cell_velocity"][cell_mask].cpu().numpy())

        # Collect data for each simulation in the batch
        snapshot_data = {}

        for j, mesh_name in enumerate(sim_ids):
            snapshot_data[mesh_name] = {
                "field_data": cell_velocity_list[j],
                "vertex_pos": v_graph_list[j].pos.cpu().numpy(),
                "vertex_face": v_graph_list[j].face.cpu().numpy()
            }
            # print(f"snapshot for {mesh_name} - vertices: {len(v_graph_list[j].pos)}, faces: {v_graph_list[j].face.shape[1]}")

        return snapshot_data

    def run(self, model, dataloader, save="full", error="on", output=None, progress=False):
        """
        Run rollout simulation.

        Args:
            model: The model to use for rollout
            dataloader: DataLoader containing initial conditions and sequential timesteps
            save: "full", "snapshot", "off" - saves each step to h5 file in rollout folder
            error: "on", "off" - computes error during rollout
            output: Output directory name
            progress: Whether to show tqdm progress bar

        Returns:
            Dictionary containing error values
        """
        model.eval()
        start_time = time.time()
        self.config.model.timestep_stride = dataloader.dataset.stride
        num_timesteps = len(dataloader)  * self.config.model.timestep_stride

        # Get initial conditions
        initial_data = next(iter(dataloader))  # Get first batch of data
        batches = [initial_data[0].batch, initial_data[1].batch, initial_data[2].batch]
        # Setup file saving
        if save == "full":
            simulation_data, simulation_file, meta_data, meta_file = self._simulation_setup(save, output, initial_data, num_timesteps, batches)

        # Main rollout loop
        model = model.to(self.device)

        # Initialise graph for rollout
        input_graphs = model.transform_features(dataloader.dataset, initial_data)
        input_graphs = [graph.to(self.device) for graph in input_graphs]

        # Initialize snapshot data collection
        snapshot_data = {}
        snapshot_indices = getattr(self.config.rollout, 'snapshot_indices', [])

        save_index = 1  # Track consecutive save indices (initial conditions are at index 0)
        with torch.no_grad():
            loss_cell_velocity = torch.zeros(num_timesteps, self.batch_size, device=self.device)
            loss_cell_pressure = torch.zeros(num_timesteps, self.batch_size,  device=self.device)
            loss_divergence = torch.zeros(num_timesteps, self.batch_size, device=self.device)
            device_batches = [batch.to(self.device) for batch in batches]

            dataloader_iter = enumerate(dataloader)
            if progress:
                dataloader_iter = tqdm(dataloader_iter, total=len(dataloader)-1)

            for batch_idx, data in dataloader_iter: # i.e. looping over timesteps

                i = batch_idx
                if i + 1 + self.data_timestep_range[0] >= self.data_timestep_range[1]:
                    break

                data = [graph.to(self.device) for graph in data]

                # Forward Pass
                output = model([graph.clone() for graph in input_graphs], mode='rollout')

                # Check if output contains multiple timesteps (k > 1)
                # Only detect multi-timestep if we have reasonable k values (2-10)
                k_steps = self.config.model.bundle_size if self.config.model.bundle_size is not None else 1

                # Process each timestep
                for k in range(k_steps):
                    if k_steps > 1:
                        # Extract k-th timestep from multi-timestep output
                        solutions = {}
                        for key, value in output.items():
                            if isinstance(value, torch.Tensor) and value.dim() >= 3:
                                # print("key: ", key)
                                # print("value: ", value.shape)
                                solutions[key] = value[:, k]
                            else:
                                solutions[key] = value
                            if key == "cell_velocity_change":
                                solutions[key] = solutions[key].squeeze(1)
                    else:
                        # Single timestep output
                        solutions = output

                    if "cell_velocity" in solutions:
                        pass
                    elif "cell_velocity_change" in solutions:
                        solutions["cell_velocity"] = input_graphs[0].x[:, 0:2] + solutions["cell_velocity_change"]

                    if "cell_pressure" in solutions:
                        pass
                    elif "face_pressure" in solutions:
                        # print(solutions["face_pressure"].unsqueeze(-1))
                        solutions["cell_pressure"] = interpolate_face_to_centroid(solutions["face_pressure"], input_graphs[1].face)

                    # error computation for timestep k
                    if error != "off":
                        self._error_accumulate(
                            i*self.config.model.timestep_stride + k, data, solutions, input_graphs, device_batches,
                            loss_cell_velocity, loss_cell_pressure, loss_divergence, target_index=k
                        )

                    # Save data for timestep k
                    if save == "full":
                        save_index = self._simulations_save(
                            simulation_data, solutions, batches, self.sim_ids, save, i * self.config.model.timestep_stride + k, save_index,
                            self.data_timestep_range, data, self.config
                        )
                    elif save == "snapshot":
                        # Collect snapshot data
                        current_timestep = i * self.config.model.timestep_stride + k + self.data_timestep_range[0]
                        if current_timestep in snapshot_indices:
                            snapshot_timestep_data = self._save_snapshot(solutions, input_graphs, current_timestep, dataloader)
                            snapshot_data[current_timestep] = snapshot_timestep_data

                # Update graph for next timestep (use last timestep's solutions)
                input_graphs = model.update_features(solutions, input_graphs)

            # Finalise Loss
            if error != "off":
                scalar_losses, evolution_arrays = self._error_save(
                    loss_cell_velocity, loss_cell_pressure, loss_divergence, dataloader
                )

        # Finalize Save
        if save == "full":
            self._simulation_cleanup(simulation_data, meta_file, meta_data, start_time)

        # Return error values
        if error != "off":
            return scalar_losses, evolution_arrays, snapshot_data
        else:
            return None, None, snapshot_data


# === MAIN ===


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()

    # Config
    with open(args.config, 'r') as f:
        data = json.load(f)
    config = Config.from_dict(data)
    device = config.settings.device

    # Checkpoint Config
    fpath = config.model.fpath
    checkpoint = torch.load(fpath, map_location=device, weights_only=False)
    train_dict = checkpoint['config']
    train_config = Config.from_dict(train_dict)
    train_config.settings = config.settings

    # Update config
    config = update_config(config, train_config)

    # Imports
    dataset_module = importlib.import_module(config.dataset.module)
    dataset_class = getattr(dataset_module, config.dataset.name)

    # Dataset
    data_directory = config.dataset.dpath
    shuffle = config.dataset.shuffle
    valid_dataset = dataset_class(data_directory, config, mode='rollout', noise=False, shuffle=shuffle, transform_fn=None)
    sim_ids = valid_dataset.get_sim_ids()

    # Model
    model = initialise_model(train_config, checkpoint, valid_dataset)
    valid_dataset.set_grad_weights(model)

    # Single-GPU dataloader
    valid_sampler = BatchSampler(
        RolloutSampler(valid_dataset),
        batch_size=config.rollout.batch_size,
        drop_last=False
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_sampler=valid_sampler,
        pin_memory=config.settings.pin_memory,
        num_workers=config.rollout.num_workers,
        persistent_workers=config.rollout.persistent_workers,
        prefetch_factor=config.rollout.prefetch_factor,
    )

    # Output
    output = config.logging.name

    # Run rollout
    rollout = Rollout(config, device, sim_ids)

    start_time = time.time()
    print("\nRollout started...")
    rollout.run(model, valid_loader, save="full", error="off", output=output, progress=True)
    print(f"\nRollout complete in t = {time.time() - start_time} s")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nRollout stopped by keyboard interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nRollout failed: {e}")
        traceback.print_exc()
        sys.exit(1)
