# === LIBRARIES ===

import h5py
import torch
import torch.nn.functional as torchFunc
import numpy as np
import time
from torch_geometric.data import Data
import enum
from torch_geometric.transforms import FaceToEdge

# === MODULES ===

import utils.geometry as custom_geom
from datasets.DataSet import DataSet

# === CLASSES ===

class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    SLIP = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    ANOTHER = 7


class DataSet_CF(DataSet):
    """
    Dataset class for cylinder flow simulations.
    Handles MeshGraphNets-style datasets with boundary node types.
    """
    def __init__(self, data_filepath, config, mode, transform_fn=None, transform=None, pre_transform=None, noise=False, shuffle=False):
        self.class_types = NodeType
        super().__init__(data_filepath, config, mode, transform_fn, transform, pre_transform, noise, shuffle)
        self.dt = 0.01

    def preprocess(self, output_fpath):
        """Convert meshgraphnets h5 dataset into model h5 format"""
        with h5py.File(self.data_filepath, 'r') as input, h5py.File(output_fpath, 'w') as output:
            # Iterate over each mesh in the input file
            print("Processing meshes...")
            start = time.time()
            mesh_limit = self.sim_limit
            timestep_limit = self.timestep_range[1]
            mesh_ids = sorted(map(int, input.keys()))
            mesh_ids = [str(mesh_id) for mesh_id in mesh_ids]
            total = self.len() # doesn't work
            if mesh_limit is not None:
                mesh_ids = mesh_ids[:mesh_limit]
                total = mesh_limit
            for i, mesh_id in enumerate(mesh_ids):
                print(f"mesh={i+1}/{total}; t={time.time() - start:.2f}s")
                data = input[mesh_id]
                mesh_group = output.create_group(f"mesh_{mesh_id}")

                ## timesteps
                num_timesteps = data['velocity'].shape[0]
                if timestep_limit is not None:
                    num_timesteps = min(num_timesteps, timestep_limit)

                ## geometry
                vertex_pos = data['pos'][0]
                vertex_type = data['node_type'][0]
                vertex_cell = np.sort(data['cells'][0], axis=1) # sorted to match original
                geom = mesh_group.create_group('geom')
                geom = self.write_geometry(geom, vertex_pos, vertex_cell, vertex_type)
                vertex_edge_index = geom['vertex_edge_index'][()]
                num_cells = vertex_cell.shape[0]
                num_faces = vertex_edge_index.shape[1]

                ## metadata
                meta = mesh_group.create_group('meta') # stores metadata such as dt
                meta["dt"] = self.dt
                meta['num_timesteps'] = num_timesteps
                meta['num_cells'] = num_cells
                meta['num_faces'] = num_faces
                meta['num_vertices'] = vertex_pos.shape[0]

                ## simulation data
                cell = mesh_group.create_group('cell')
                face = mesh_group.create_group('face')

                cell.create_dataset('velocity',  shape=(num_timesteps, num_cells, 2),
                                    dtype='f4', chunks=(1, num_cells, 2),
                                    compression='gzip', compression_opts=4)
                cell.create_dataset('pressure',  shape=(num_timesteps, num_cells, 1),
                                    dtype='f4', chunks=(1, num_cells, 1),
                                    compression='gzip', compression_opts=4)
                face.create_dataset('velocity',  shape=(num_timesteps, num_faces, 2),
                                    dtype='f4', chunks=(1, num_faces, 2),
                                    compression='gzip', compression_opts=4)
                face.create_dataset('pressure',  shape=(num_timesteps, num_faces, 1),
                                    dtype='f4', chunks=(1, num_faces, 1),
                                    compression='gzip', compression_opts=4)

                for ts in range(num_timesteps):
                    print(f"  ts={ts+1}/{num_timesteps}", end='\r', flush=True)
                    velocity_vertex  = np.asarray(data['velocity'][ts])      # (Nv, 2)
                    pressure_vertex  = np.asarray(data['pressure'][ts])      # (Nv, 1)

                    v_cell  = custom_geom.interpolate_centroid(velocity_vertex, vertex_cell, geom["cell_pos"])    # (Nc, 2)
                    p_cell  = custom_geom.interpolate_centroid(pressure_vertex,  vertex_cell, geom["cell_pos"])    # (Nc, 1)
                    v_face = (velocity_vertex[vertex_edge_index[0]] + velocity_vertex[vertex_edge_index[1]]) / 2.0  # (Nf, 2)
                    p_face = (pressure_vertex[vertex_edge_index[0]] + pressure_vertex[vertex_edge_index[1]]) / 2.0  # (Nf, 1)

                    cell['velocity'][ts] = v_cell.astype('f8')
                    cell['pressure'][ts] = p_cell.astype('f8')
                    face['velocity'][ts] = v_face.astype('f8')
                    face['pressure'][ts] = p_face.astype('f8')

        return


if __name__ == '__main__':
    pass
