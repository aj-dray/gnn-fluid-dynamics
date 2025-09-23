"""Functions and classes for saving graph data for Rollout"""
# === LIBRARIES ===


import h5py
import os
import torch
from torch_geometric.utils import unbatch
from torch_geometric.data import Batch


# === MODULES ===


import utils.geometry


# === FUNCTIONS ===


def interpolate_face_to_cell(face_values, face_indices):

    interpolated_values = torch.zeros(face_indices.shape[0], device=face_values.device)
    gathered_face_values = face_values[face_indices]  # [num_cells, 3]
    interpolated_values = torch.mean(gathered_face_values, dim=0)

    return interpolated_values.reshape(-1, 1)  # Convert to column vector


# === CLASSES ===


class SimulationData():
    def __init__(self, file_obj, graphs, sim_ids, num_timesteps):
        """Object to save rollout data"""
        if isinstance(file_obj, str):
            self.file = h5py.File(file_obj, 'w')
            self.close_file = True
        else:
            self.file = file_obj
            self.close_file = False

        c_graph, f_graph, v_graph = graphs

        # Unbatch the graphs to get individual samples
        c_graph_list = Batch.to_data_list(c_graph)
        f_graph_list = Batch.to_data_list(f_graph)
        v_graph_list = Batch.to_data_list(v_graph)

        ## Mesh Groups
        for j, id in enumerate(sim_ids):
            mesh_group = self.file.create_group(id)
            # print(f"Loop is at mesh {id}.")
            # print(f"Actual examples is {c_graph_list[j].mesh_id}")
            ## Metadata
            self.meta = mesh_group.create_group('meta')  # stores metadata such as dt
            num_cells = c_graph_list[j].pos.shape[0]
            self.meta["num_cells"] = num_cells
            num_faces = f_graph_list[j].pos.shape[0]
            self.meta["num_faces"] = num_faces
            self.meta.create_dataset('timestep', shape=(num_timesteps,),
                dtype='f4', chunks=(1,),
                compression='gzip', compression_opts=4)

            # Store face index for each simulation
            if not hasattr(self, 'face_indices'):
                self.face_indices = {}
            self.face_indices[id] = f_graph_list[j].face

            ## Geometry
            geom = mesh_group.create_group('geom')
            geom.create_dataset('vertex_pos', data=v_graph_list[j].pos.cpu().numpy())
            geom.create_dataset('vertex_edge_index', data=v_graph_list[j].edge_index.cpu().numpy())
            geom.create_dataset('vertex_face', data=v_graph_list[j].face.cpu().numpy())
            geom.create_dataset('face_face', data=f_graph_list[j].face.cpu().numpy())
            geom.create_dataset('cell_pos', data=c_graph_list[j].pos.cpu().numpy())
            geom.create_dataset('cell_normal', data=c_graph_list[j].normal.cpu().numpy())
            geom.create_dataset('cell_volume', data=c_graph_list[j].volume.cpu().numpy())
            geom.create_dataset('cell_edge_index', data=c_graph_list[j].edge_index.cpu().numpy())
            geom.create_dataset('face_type', data=f_graph_list[j].type.cpu().numpy())
            geom.create_dataset('face_normal', data=f_graph_list[j].normal.cpu().numpy())
            geom.create_dataset('face_area', data=f_graph_list[j].area.cpu().numpy())
            geom.create_dataset('face_pos', data=f_graph_list[j].pos.cpu().numpy())
            geom.create_dataset('boundary_mask', data=f_graph_list[j].boundary_mask.cpu().numpy())

            ## Data
            cell = mesh_group.create_group('cell')
            cell.create_dataset('velocity', shape=(num_timesteps, num_cells, 2),
                        dtype='f4', chunks=(1, num_cells, 2),
                        compression='gzip', compression_opts=4)
            cell.create_dataset('pressure', shape=(num_timesteps, num_cells, 1),
                        dtype='f4', chunks=(1, num_cells, 1),
                        compression='gzip', compression_opts=4)
            cell.create_dataset('flux', shape=(num_timesteps, num_cells, 3),
                        dtype='f4', chunks=(1, num_cells, 3),
                        compression='gzip', compression_opts=4)
            # Ground truth datasets
            cell.create_dataset('velocity_gt', shape=(num_timesteps, num_cells, 2),
                        dtype='f4', chunks=(1, num_cells, 2),
                        compression='gzip', compression_opts=4)
            cell.create_dataset('pressure_gt', shape=(num_timesteps, num_cells, 1),
                        dtype='f4', chunks=(1, num_cells, 1),
                        compression='gzip', compression_opts=4)

            face = mesh_group.create_group('face')
            face.create_dataset('velocity', shape=(num_timesteps, num_faces, 2),
                    dtype='f4', chunks=(1, num_faces, 2),
                    compression='gzip', compression_opts=4)
            face.create_dataset('pressure', shape=(num_timesteps, num_faces, 1),
                    dtype='f4', chunks=(1, num_faces, 1),
                    compression='gzip', compression_opts=4)
            face.create_dataset('flux', shape=(num_timesteps, num_faces, 1),
                    dtype='f4', chunks=(1, num_faces, 1),
                    compression='gzip', compression_opts=4)
            # Ground truth datasets
            face.create_dataset('velocity_gt', shape=(num_timesteps, num_faces, 2),
                    dtype='f4', chunks=(1, num_faces, 2),
                    compression='gzip', compression_opts=4)
            face.create_dataset('pressure_gt', shape=(num_timesteps, num_faces, 1),
                    dtype='f4', chunks=(1, num_faces, 1),
                    compression='gzip', compression_opts=4)

    def save_timestep(self, solutions, batches, sim_ids, index, timestep, ground_truth=None):
        cell_batch, face_batch, vertex_batch = batches

        # Transfer all tensors to CPU first to batch GPU->CPU transfers
        cpu_solutions = {}
        for key, value in solutions.items():
            cpu_solutions[key] = value.cpu()

        cpu_ground_truth = {}
        if ground_truth is not None:
            # print(ground_truth['dt'])
            for key, value in ground_truth.items():
                # print(f"Saving ground truth {key} with shape {value.shape}")
                cpu_ground_truth[key] = value.cpu()

        # Unbatch solutions based on their type
        unbatched_solutions = {}
        for key, value in cpu_solutions.items():
            if key.startswith('cell_'):
                unbatched_solutions[key] = unbatch(value, cell_batch)
            elif key.startswith('face_'):
                unbatched_solutions[key] = unbatch(value, face_batch)
            elif key.startswith('vertex_'):
                unbatched_solutions[key] = unbatch(value, vertex_batch)

        # Unbatch ground truth data if provided
        unbatched_ground_truth = {}
        if ground_truth is not None:
            for key, value in cpu_ground_truth.items():
                if key.startswith('cell_'):
                    unbatched_ground_truth[key] = unbatch(value, cell_batch)
                elif key.startswith('face_'):
                    unbatched_ground_truth[key] = unbatch(value, face_batch)
                elif key.startswith('vertex_'):
                    unbatched_ground_truth[key] = unbatch(value, vertex_batch)

        for j, id in enumerate(sim_ids):
            self.file[id]['meta']['timestep'][index] = int(timestep)

            # Save cell data
            if 'cell_velocity' in unbatched_solutions:
                self.file[id]['cell']['velocity'][index] = unbatched_solutions['cell_velocity'][j].numpy().astype('f4')

            if 'cell_pressure' in unbatched_solutions:
                self.file[id]['cell']['pressure'][index] = unbatched_solutions['cell_pressure'][j].numpy().astype('f4')

            if 'cell_flux' in unbatched_solutions:
                self.file[id]['cell']['flux'][index] = unbatched_solutions['cell_flux'][j].numpy().astype('f4')

            # Save face data
            if 'face_velocity' in unbatched_solutions:
                self.file[id]['face']['velocity'][index] = unbatched_solutions['face_velocity'][j].numpy().astype('f4')

            if 'face_pressure' in unbatched_solutions:
                self.file[id]['face']['pressure'][index] = unbatched_solutions['face_pressure'][j].numpy().astype('f4')

                # Also interpolate face pressure to cell pressure if cell_pressure not already provided
                if 'cell_pressure' not in unbatched_solutions:
                    face_index = self.face_indices[id]
                    cell_pressure = utils.geometry.interpolate_face_to_centroid(unbatched_solutions['face_pressure'][j], face_index)
                    self.file[id]['cell']['pressure'][index] = cell_pressure.numpy().astype('f4')

            if 'face_flux' in unbatched_solutions:
                self.file[id]['face']['flux'][index] = unbatched_solutions['face_flux'][j].numpy().astype('f4')

            # Save ground truth data
            if ground_truth is not None:
                if 'cell_velocity' in unbatched_ground_truth:
                    self.file[id]['cell']['velocity_gt'][index] = unbatched_ground_truth['cell_velocity'][j].numpy().astype('f4')

                if 'cell_pressure' in unbatched_ground_truth:
                    self.file[id]['cell']['pressure_gt'][index] = unbatched_ground_truth['cell_pressure'][j].numpy().astype('f4')

                if 'face_velocity' in unbatched_ground_truth:
                    self.file[id]['face']['velocity_gt'][index] = unbatched_ground_truth['face_velocity'][j].numpy().astype('f4')

                if 'face_pressure' in unbatched_ground_truth:
                    self.file[id]['face']['pressure_gt'][index] = unbatched_ground_truth['face_pressure'][j].numpy().astype('f4')

                # # If 'dt' is not present in meta, add it from unbatched_ground_truth[0].dt
                # if 'dt' in unbatched_ground_truth and 'dt' not in self.file[id]['meta']:
                #     print(f"Adding dt {unbatched_ground_truth['dt'][j].item()}")
                #     self.file[id]['meta']['dt'] = unbatched_ground_truth['dt'][j].item()


    def close(self):
        """Close the HDF5 file when done, but only if we created it"""
        if self.close_file:
            self.file.close()
