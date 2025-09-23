# === LIBRARIES ---

import h5py
import numpy as np
import os
import pyvista as pv
import re
from scipy.spatial import cKDTree
import enum

# === MODULES ---

import utils.geometry as custom_geom
from datasets.DataSet import DataSet_FVGN

# === CLASSES ---

class NodeType(enum.IntEnum):
    NORMAL = 0
    PERIODIC = 1


class DataSet_FVGN_TaylorGreen(DataSet_FVGN):
    """Variant of dataset for fvgn"""
    def __init__(self, data_filepath, config, mode, transform=None, pre_transform=None, noise=False, shuffle=False):
        super().__init__(data_filepath, config, mode, transform, pre_transform, noise, shuffle)
        self.norm = True
        self.node_feature_size = 2 + len(NodeType)
        self.edge_feature_size = 5 + len(NodeType)
        self.dt = None
        self.class_types = NodeType

    @staticmethod
    def numeric_sort(names):
        key = lambda s: float(re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)[-1])
        return sorted(names, key=key)

    def write_geometry(self, geom, vertex_pos, vertex_cell, vertex_type):
        # face_index, cell_edge_index, vertex_edge_index = custom_geom.compute_cell_face_connectivity(vertex_cell)

        # vertex_edge_vector = vertex_pos[vertex_edge_index[1], :] -  vertex_pos[vertex_edge_index[0], :]
        # face_area = np.linalg.norm(vertex_edge_vector, axis=1).reshape(-1, 1)
        # face_pos = np.mean(vertex_pos[vertex_edge_index.T], axis=1)

        cell_pos = np.mean(vertex_pos[vertex_cell], axis=1)
        cell_volume = custom_geom.compute_cell_volume(vertex_pos, vertex_cell).reshape(-1, 1)

        # normal = np.stack((-vertex_edge_vector[:, 1], vertex_edge_vector[:, 0]), axis=1)
        # norm = np.linalg.norm(normal, axis=1, keepdims=True)
        # face_normal = normal / (norm + 1e-8)
        # face_normal = custom_geom.correct_normals(cell_pos, cell_edge_index, face_normal, face_pos) # ensure owner -> neighbour
        # face_type = custom_geom.classify_edges(vertex_edge_index, vertex_type, cell_edge_index, self.class_types)
        # face_boundary_mask = cell_edge_index[0] == cell_edge_index[1]

        # cell_type = custom_geom.classify_cells(face_index, face_type, self.class_types)

        geom.create_dataset('vertex_pos', data=vertex_pos)
        # geom.create_dataset('vertex_edge_index', data=vertex_edge_index)
        geom.create_dataset('vertex_face', data=vertex_cell.T)
        # geom.create_dataset('vertex_edge_vector', data=vertex_edge_vector)
        # geom.create_dataset('face_normal', data=face_normal)
        # geom.create_dataset('face_pos', data=face_pos)
        # geom.create_dataset('face_area', data=face_area)
        # geom.create_dataset('face_index', data=face_index.T)
        # geom.create_dataset('face_type', data=face_type)
        # geom.create_dataset('face_boundary_mask', data=face_boundary_mask)
        geom.create_dataset('cell_pos', data=cell_pos)
        # geom.create_dataset('cell_edge_index', data=cell_edge_index)
        geom.create_dataset('cell_volume', data=cell_volume)
        # geom.create_dataset('cell_type', data=cell_type)

        return geom


    def preprocess(self, output_fpath):
        self.data_filepath = os.path.dirname(self.data_filepath) + '/' + self.mode
        with h5py.File(output_fpath, 'w') as output:
            # Simulations
            sim_folders = [d for d in os.listdir(self.data_filepath) if os.path.isdir(os.path.join(self.data_filepath, d))]
            num_sim_folders = len(sim_folders)
            if self.sim_limit is not None:
                print("simlimit", self.sim_limit)
                num_sim_folders = min(num_sim_folders, self.sim_limit)
                sim_folders = sim_folders[:num_sim_folders]

            print(f"Processing {num_sim_folders} {self.mode} meshes:")
            for i, vtk_directory in enumerate(sim_folders):
                print(f"\tmesh {i+1}/{num_sim_folders}: {vtk_directory}")
                full_fpath = os.path.join(self.data_filepath, vtk_directory)
                mesh_name = os.path.basename(vtk_directory)
                mesh_group = output.create_group(mesh_name)

                ## timesteps
                ts_dirs = [d for d in os.listdir(os.path.join(self.data_filepath, vtk_directory))
                                            if os.path.isdir(os.path.join(self.data_filepath, vtk_directory, d))]
                ts_dirs = self.numeric_sort(ts_dirs)
                timestep_start = self.config.dataset.get("timestep_start", 0)
                if self.config.dataset.get("timestep_limit"):
                    ts_dirs = ts_dirs[timestep_start:self.config.dataset["timestep_limit"]]
                else:
                    ts_dirs = ts_dirs[timestep_start:]
                num_timesteps = len(ts_dirs)
                print(f"\t\t{num_timesteps} timesteps found")

                ## geometry
                # print(ts_dirs)
                first_timestep = f"{full_fpath}/{ts_dirs[0]}.vtm"
                data = pv.read(first_timestep)
                mesh3D = data['internal']

                bounds = data.bounds # take 2d slice of 3d mesh
                z_mid = 0.5 * (bounds[4] + bounds[5])
                mesh = mesh3D.slice(normal='z', origin=(0, 0, z_mid))
                vertex_pos = mesh.points[:, :2]

                n_points = vertex_pos.shape[0] # combine boundary and internal points
                point_labels = np.full(n_points, "internal", dtype=object)
                tree = cKDTree(vertex_pos)
                patch_files = {
                    "bottom": "boundary/bottom.vtp",
                    "left": "boundary/left.vtp",
                    "right": "boundary/right.vtp",
                    "top": "boundary/top.vtp"
                }
                for label, path in patch_files.items():
                    # print(f"Processing {label} patch")
                    fpath = os.path.join(full_fpath, ts_dirs[0], path)
                    patch = pv.read(fpath)
                    patch = patch.slice(normal='z', origin=(0, 0, z_mid))
                    patch_points = patch.points[:, :2]
                    matched_indices = tree.query(patch_points, k=1)[1]
                    point_labels[matched_indices] = label
                vertex_type = np.full(n_points, NodeType.NORMAL, dtype=np.int32)
                vertex_type[np.isin(point_labels, list(patch_files.keys()))] = NodeType.PERIODIC


                cells = []
                for i in range(mesh.n_cells):
                    cell = mesh.get_cell(i)
                    cell_points = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
                    cells.append(cell_points)
                vertex_cell = np.array(cells, dtype=np.int32)

                geom = mesh_group.create_group('geom')
                print(vertex_cell.shape)
                geom = self.write_geometry(geom, vertex_pos, vertex_cell, vertex_type)
                # vertex_edge_index = geom['vertex_edge_index'][()]
                # cell_edge_index = geom['cell_edge_index'][()]
                num_cells = vertex_cell.shape[0]
                # num_faces = vertex_edge_index.shape[1]
                # face_pos   = geom['face_pos'][()]        # (nF, 2) centroids in xy
                # face_kdtree = cKDTree(face_pos)

                # patch_to_face = {}                       # dict: patch_name -> 1-D array of face indices

                # for label, rel_path in patch_files.items():          # inlet, outlet, walls, obstacle
                #     patch_path = os.path.join(full_fpath, ts_dirs[0], rel_path)
                #     patch      = pv.read(patch_path).slice(normal='z', origin=(0, 0, z_mid))

                #     # we want a single point per patch *cell* → use the cell centres
                #     patch_centroids = patch.cell_centers().points[:, :2]

                #     # nearest global face for every patch face
                #     idx = face_kdtree.query(patch_centroids, k=1)[1]
                #     patch_to_face[label] = np.unique(idx)            # store unique indices

                ## simulation data
                cell = mesh_group.create_group('cell')
                # face = mesh_group.create_group('face')

                cell.create_dataset('velocity',  shape=(num_timesteps, num_cells, 2),
                                    dtype='f4', chunks=(1, num_cells, 2),
                                    compression='gzip', compression_opts=4)
                cell.create_dataset('pressure',  shape=(num_timesteps, num_cells, 1),
                                    dtype='f4', chunks=(1, num_cells, 1),
                                    compression='gzip', compression_opts=4)
                # face.create_dataset('velocity',  shape=(num_timesteps, num_faces, 2),
                #                     dtype='f4', chunks=(1, num_faces, 2),
                #                     compression='gzip', compression_opts=4)
                # face.create_dataset('pressure',  shape=(num_timesteps, num_faces, 1),
                #                     dtype='f4', chunks=(1, num_faces, 1),
                #                     compression='gzip', compression_opts=4)

                # Process each timestep
                for ts, ts_dir in enumerate(ts_dirs):
                    print(f"\t\ttimestep {ts+1}/{num_timesteps}", end='\r', flush=True)
                    # Read the VTM file for this timestep
                    timestep_path = f"{full_fpath}/{ts_dir}.vtm"
                    timestep_data = pv.read(timestep_path)
                    mesh3D = timestep_data['internal']
                    mesh = mesh3D.slice(normal='z', origin=(0, 0, z_mid))

                    cell_velocity = mesh.cell_data['U'][:, :2]  # Keep only x,y components
                    cell_pressure = mesh.cell_data['p'].reshape(-1, 1)

                    # cell_pos = geom['cell_pos'][()]
                    # face_pos = geom['face_pos'][()]
                    # face_velocity = custom_geom.cell_to_face(cell_velocity, cell_edge_index, face_pos, cell_pos)
                    # face_pressure = custom_geom.cell_to_face(cell_pressure, cell_edge_index, face_pos, cell_pos)

                    # for patch_name, face_idx in patch_to_face.items():
                    #     bc_path = os.path.join(full_fpath, ts_dir, f"boundary/{patch_name}.vtp")
                    #     bc_mesh = pv.read(bc_path).slice(normal='z', origin=(0, 0, z_mid))

                    #     # The fields may be stored either on points or on cells – handle both
                    #     if 'U' in bc_mesh.cell_data.keys():
                    #         U_patch = bc_mesh.cell_data['U'][:, :2]          # shape (n_patch_cells, 2)
                    #         p_patch = bc_mesh.cell_data['p'].reshape(-1, 1)

                    #     face_velocity[face_idx] = U_patch
                    #     face_pressure[face_idx] = p_patch

                    # Store data for this timestep
                    cell['velocity'][ts] = cell_velocity.astype('f4')
                    cell['pressure'][ts] = cell_pressure.astype('f4')
                    # face['velocity'][ts] = face_velocity.astype('f4')
                    # face['pressure'][ts] = face_pressure.astype('f4')
                print('\n')

        return
