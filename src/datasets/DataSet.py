# === LIBRARIES ===


from torch_geometric.data import Dataset
import torch
import h5py
import json
from collections import OrderedDict
import numpy as np
import tqdm
from importlib import import_module
from torch_geometric.loader import DataLoader
import os


# === MODULES ===


import utils.geometry as custom_geom
from torch_geometric.data import Data
from utils.normalisation import CustomAccumulator


# === CLASSES ===


class DataSet(Dataset):
    """
    Base dataset class for simulation data handling.
    Child classes construct graphs specific to model requirements.
    
    Args:
        data_filepath: Path to HDF5 data files
        config: Configuration object with dataset parameters
        mode: Dataset mode ('train', 'rollout', 'preproc', 'stats')
        transform_fn: Model-specific feature transformation function
        transform: PyTorch Geometric transform
        pre_transform: PyTorch Geometric pre-transform
        noise: Whether to apply noise augmentation
        shuffle: Whether to shuffle simulation order
    """
    def __init__(self, data_filepath, config, mode, transform_fn=None, transform=None, pre_transform=None, noise=False, shuffle=False):
        self.mode = mode # i.e. train, valid, preproc
        self.config = config

        if self.mode == 'train':
            self.data_subset = config.training.data_subset
            self.sim_limit = config.training.data_sim_limit
            self.timestep_range =  config.training.data_timestep_range
        elif self.mode == 'rollout':
            self.data_subset = config.rollout.data_subset
            self.sim_limit = config.rollout.data_sim_limit
            self.timestep_range =  config.rollout.data_timestep_range
        elif self.mode == 'preproc' or self.mode == 'stats':
            self.data_subset = config.preproc.data_subset
            self.sim_limit = config.preproc.data_sim_limit
            self.timestep_range =  config.preproc.data_timestep_range

        self.data_filepath = data_filepath + '/' + self.data_subset + '.h5'
        self.sample_map = []
        self.shuffle = shuffle
        self.file_handler = None
        self._geom_cache = OrderedDict()
        self.max_cached_meshes = 25
        self.transform_fn = transform_fn
        self.noise = noise
        self.stats = None # applied once found
        self.cell_grad_weights_use = False
        self.face_grad_weights_use = False

        if self.config.model.timestep_stride:
            self.stride = self.config.model.timestep_stride
            self.data_window = self.config.model.timestep_stride + 1
        else:
            self.stride = 1
            self.data_window = 2
        if self.config.training.pushforward_factor:
            self.stride = 1
            self.data_window = self.config.training.pushforward_factor + 2


        if self.config.model.bundle_size:
            print("Warning: temporal bundling running: make sure model matches.")
            self.data_window = self.config.model.bundle_size + 1
            if self.mode == "rollout":
                self.stride = self.config.model.bundle_size

        print("stride: ", self.stride)
        print("data_window: ", self.data_window)


        if not self.mode == 'preproc':
            self._create_map()

        super().__init__(None, transform, pre_transform)

        print(f"\nDataset setup for {self.mode} using {self.data_filepath}.")

    def _create_map(self):
        """Create mapping of available simulation timesteps for data loading."""
        with h5py.File(self.data_filepath, 'r', swmr=True) as f:
            group_ids = np.array([k for k in f.keys() if k.startswith('mesh')])
            num_timesteps = f[group_ids[0]]['meta']['num_timesteps'][()]
            if self.shuffle:
                np.random.shuffle(group_ids)

            if self.mode == 'rollout' and self.config.rollout.data_sim_index:
                self.group_ids = [f"mesh_{i}" for i in self.config.rollout.data_sim_index]
            elif self.sim_limit:
                assert len(group_ids) >= self.sim_limit
                self.group_ids = group_ids[:self.sim_limit]
            else:
                self.group_ids = group_ids

            if self.timestep_range:
                # print(self.data_window)
                # print(self.timestep_range[1] - 2 + self.data_window)
                assert num_timesteps >= self.timestep_range[1] - 2 + self.data_window
                start, end = self.timestep_range[:2]
            else:
                start, end = [0, num_timesteps]

            for ts in range(start, end, self.stride):
                for g_id in self.group_ids:
                    self.sample_map.append((g_id, ts))

    def _initialise(self, mode='r'):
        # Always reinitialize in worker processes to avoid sharing file handles
        if mode == 'r':
            if self.file_handler is None or not self.file_handler.id.valid:
                if self.file_handler is not None:
                    try:
                        self.file_handler.close()
                    except:
                        pass
                self.file_handler = h5py.File(self.data_filepath, 'r', swmr=True)
        elif mode == 'a':
            if self.file_handler is not None:
                try:
                    self.file_handler.close()
                except:
                    pass
            self.file_handler = h5py.File(self.data_filepath, 'a')
            # self.file_handler.swmr_mode = True  # <-- Enable SWMR write mode

    def _close(self):
        if self.file_handler is not None:
            self.file_handler.close()
            self.file_handler = None

    def get(self, idx): # called when indexing dataset
        self._initialise()
        mesh_id, ts = self.sample_map[idx]
        # print(f"Loading mesh {mesh_id} at timestep {ts}")
        geom = self._get_geom(mesh_id) # with caching
        graphs = self._process_timestep(self.file_handler[mesh_id], ts, geom, mesh_id)
        if self.transform_fn is not None:
            graphs = self.transform_fn(self, graphs)
        return graphs

    def _get_geom(self, mesh_id):
        if mesh_id in self._geom_cache:
            geom = self._geom_cache.pop(mesh_id)
            self._geom_cache[mesh_id] = geom
        else:
            geom = self._load_geom(mesh_id)
            self._geom_cache[mesh_id] = geom

            if (self.max_cached_meshes is not None and
                len(self._geom_cache) > self.max_cached_meshes):
                self._geom_cache.popitem(last=False)
        return geom

    def _load_geom(self, mesh_id):
        g = self.file_handler[mesh_id]["geom"]
        return {name: g[name][()] for name in g.keys()}

    def len(self): # needed for base class
        return len(self.sample_map)

    def __del__(self): # called when goes out of scope
        try:
            if hasattr(self, 'file_handler') and self.file_handler is not None:
                self.file_handler.close()
        except Exception:
            pass

    def get_sim_ids(self):
        return self.group_ids

    def select_trajectory(self, trajectory_id, timestep=0):
        with h5py.File(self.data_filepath, 'r', swmr=True) as f:
            mesh_id = f"mesh_{trajectory_id}"
            if mesh_id not in f:
                raise Exception(f"Trajectory {mesh_id} not found")
            # Temporarily set file handler for geometry loading
            old_handler = self.file_handler
            self.file_handler = f
            geom = self._get_geom(mesh_id)
            graphs = self._process_timestep(f[mesh_id], timestep, geom)
            self.file_handler = old_handler
            return graphs

    def _add_noise(self, velocity):
        noise_std = self.config.training.noise_std
        velocity_noise = torch.normal(mean=0.0, std=noise_std, size=velocity.shape, dtype=velocity.dtype, device=velocity.device)
        velocity += velocity_noise
        return velocity

    def _process_timestep(self, data, ts, geom, mesh_id=None):
        """Present data in graph format but features and targets not created."""
        dftype = torch.float
        ditype = torch.long

        meta_data = data['meta']
        face_data = data['face']
        cell_data = data['cell']

        dt = torch.tensor(meta_data['dt'][()], dtype=dftype)
        velocity_data = torch.tensor(cell_data['velocity'][ts:ts+(self.data_window)], dtype=dftype).transpose(0, 1)
        pressure_data = torch.tensor(cell_data['pressure'][ts:ts+(self.data_window)], dtype=dftype).transpose(0, 1)

        c_graph_dict = {
            'pos': torch.tensor(geom['cell_pos'], dtype=dftype),
            'volume': torch.tensor(geom['cell_volume'], dtype=dftype),
            'edge_index': torch.tensor(geom['cell_edge_index'], dtype=ditype),
            'normal': torch.tensor(geom['cell_normal'], dtype=dftype),
            'velocity': velocity_data,
            'pressure': pressure_data,
            'dt': dt * self.stride
        }
        if 'Re' in meta_data:
            c_graph_dict['Re'] = torch.tensor(meta_data['Re'][()], dtype=dftype)
        if mesh_id:
            c_graph_dict['mesh_id'] = mesh_id
        if self.cell_grad_weights_use and 'cell_grad_weights' in data:
            datum = data['cell_grad_weights'][str(self.config.model.cell_grad_weights_order)]
            c_graph_dict['grad_weights'] = torch.tensor(datum['weights'][()], dtype=dftype)
            c_graph_dict['grad_neighbours'] = torch.tensor(datum['neighbours'][()], dtype=ditype)

        c_graph = Data(**c_graph_dict)

        # Build face graph - dynamically add available data
        face_velocity_data = torch.tensor(face_data['velocity'][ts:ts+(self.data_window)], dtype=dftype).transpose(0, 1)
        face_pressure_data = torch.tensor(face_data['pressure'][ts:ts+(self.data_window)], dtype=dftype).transpose(0, 1)

        f_graph_dict = {
            'pos': torch.tensor(geom['face_pos'], dtype=dftype),
            'face': torch.tensor(geom["face_index"], dtype=ditype),
            'type': torch.tensor(geom["face_type"], dtype=ditype),
            'num_types': len(self.class_types),
            'area': torch.tensor(geom["face_area"], dtype=dftype),
            'boundary_mask': torch.tensor(geom["face_boundary_mask"], dtype=torch.bool),
            'normal': torch.tensor(geom['face_normal'], dtype=dftype),
            'velocity': face_velocity_data,
            'pressure': face_pressure_data,
        }
        if 'flux' in face_data:
            flux_data = torch.tensor(face_data['flux'][ts:ts+(self.data_window)], dtype=dftype).transpose(0, 1) / 0.001
            f_graph_dict['flux'] = flux_data
        if self.face_grad_weights_use and 'face_grad_weights' in data:
            datum = data['face_grad_weights'][str(self.config.model.face_grad_weights_order)]
            f_graph_dict['grad_weights'] = torch.tensor(datum['weights'][()], dtype=dftype)
            f_graph_dict['grad_neighbours'] = torch.tensor(datum['neighbours'][()], dtype=ditype)

        f_graph = Data(**f_graph_dict)

        v_graph = Data(
            pos=torch.tensor(geom['vertex_pos'], dtype=dftype),
            edge_index=torch.tensor(geom['vertex_edge_index'], dtype=ditype),
            face=torch.tensor(geom['vertex_face'], dtype=ditype),
        )

        return [c_graph, f_graph, v_graph]

    def write_geometry(self, geom, vertex_pos, vertex_cell, vertex_type):
        face_index, cell_edge_index, vertex_edge_index = custom_geom.compute_connectivity(vertex_cell, vertex_pos)

        vertex_edge_vector = vertex_pos[vertex_edge_index[1], :] -  vertex_pos[vertex_edge_index[0], :]
        face_area = np.linalg.norm(vertex_edge_vector, axis=1).reshape(-1, 1)
        face_pos = np.mean(vertex_pos[vertex_edge_index.T], axis=1)

        cell_pos = np.mean(vertex_pos[vertex_cell], axis=1)
        cell_volume = custom_geom.compute_cell_volume(vertex_pos, vertex_cell).reshape(-1, 1)

        normal = np.stack((-vertex_edge_vector[:, 1], vertex_edge_vector[:, 0]), axis=1)
        norm = np.linalg.norm(normal, axis=1, keepdims=True)
        face_normal = normal / (norm + 1e-8)
        face_normal = custom_geom.correct_normals(cell_pos, cell_edge_index, face_normal, face_pos) # ensure owner -> neighbour
        face_type = custom_geom.classify_edges(vertex_edge_index, vertex_type, cell_edge_index, self.class_types)
        face_boundary_mask = cell_edge_index[0] == cell_edge_index[1]

        cell_normal = custom_geom.compute_cell_normal(cell_pos, face_index, face_normal, face_pos)
        # cell_type = custom_geom.classify_cells(face_index, face_type, self.class_types)

        geom.create_dataset('vertex_pos', data=vertex_pos)
        geom.create_dataset('vertex_edge_index', data=vertex_edge_index)
        geom.create_dataset('vertex_face', data=vertex_cell.T)
        geom.create_dataset('vertex_edge_vector', data=vertex_edge_vector)
        geom.create_dataset('face_normal', data=face_normal)
        geom.create_dataset('face_pos', data=face_pos)
        geom.create_dataset('face_area', data=face_area)
        geom.create_dataset('face_index', data=face_index)
        geom.create_dataset('face_type', data=face_type)
        geom.create_dataset('face_boundary_mask', data=face_boundary_mask)
        geom.create_dataset('cell_pos', data=cell_pos)
        geom.create_dataset('cell_edge_index', data=cell_edge_index)
        geom.create_dataset('cell_volume', data=cell_volume)
        # geom.create_dataset('cell_type', data=cell_type)
        geom.create_dataset('cell_normal', data=cell_normal)

        return geom

    def read_stats(self, stats_fpath, model_class):
        device = self.config.settings.device
        use_cuda = device == 'cuda' and torch.cuda.is_available()

        registry, input_map, output_map = model_class.get_normalisation_map()
        accumulator = CustomAccumulator(registry, input_map, output_map, device, stats_fpath=stats_fpath)
        if not self.config.dataset.stats_recompute:
            is_complete = accumulator.check_existing()
            if is_complete:
                return accumulator.get_stats()
        # Create dataloader for efficient batched processing
        dataloader = DataLoader(
            self,
            batch_size=self.config.training.batch_size,
            shuffle=False,  # No need to shuffle for stats
            num_workers=self.config.training.num_workers,
            pin_memory=use_cuda,
            persistent_workers=False  # Simpler for stats computation
        )
        accumulator.run(dataloader, stats_recompute=self.config.dataset.stats_recompute)
        stats = accumulator.get_stats()
        accumulator.save_stats()

        return stats

    def set_noise_std(self, stats):
        if not self.config.training.noise_std and self.config.training.noise_std != 0.0:
            self.config.training.noise_std = self.config.training.noise_std_norm * stats['cell_velocity_x']["mean"]
        print("Noise std set to:", self.config.training.noise_std)

    def set_grad_weights(self, model):
        print(model.face_grad_weights_use)
        if model.cell_grad_weights_use: # for divergence-free vector field
            self.cell_grad_weights_use = True
            model.cell_mls_weights.add_weights_to_dataset(self, self.config.dataset.grad_weights_recompute)
        if model.face_grad_weights_use: # for real space integration
            self.face_grad_weights_use = True
            print("yes")
            model.face_mls_weights.add_weights_to_dataset(self, self.config.dataset.grad_weights_recompute)
