# === LIBRARIES ---

import h5py
import numpy as np
import os
import pyvista as pv
import re
from scipy.spatial import cKDTree
import enum
import pandas as pd

# === MODULES ---

import utils.geometry as custom_geom
from datasets.DataSet import DataSet

# === CLASSES ---

class NodeType(enum.IntEnum):
    NORMAL = 0
    INFLOW = 2
    OUTFLOW = 3
    WALL_BOUNDARY = 1
    SLIP = 4


class DataSet_OF(DataSet):
    """
    Dataset class for OpenFOAM VTK simulation data.
    Converts finite volume mesh data for neural network training.
    """
    def __init__(self, data_filepath, config, mode, transform_fn=None, transform=None, pre_transform=None, noise=False, shuffle=False):
        self.class_types = NodeType
        super().__init__(data_filepath, config, mode, transform_fn, transform, pre_transform, noise, shuffle)
        self.dt = None # selected from metadata file

    @staticmethod
    def numeric_sort(names):
        def key(s):
            numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
            return float(numbers[-1]) if numbers else float('inf')
        return sorted(names, key=key)

    def preprocess(self, output_fpath):
        """
        Convert OpenFOAM VTK simulation data into HDF5 format for model training.
        
        Args:
            output_fpath: Path for output HDF5 file
        """
        vtk_data_path = self.data_filepath.replace('h5', 'vtk')
        vtk_data_path = vtk_data_path.split(".")[0]
        self.data_filepath = vtk_data_path
        print("VTK: ", self.data_filepath)
        os.makedirs(os.path.dirname(output_fpath), exist_ok=True)
        print("H5: ", output_fpath)
        with h5py.File(output_fpath, 'a') as output:
            # Simulations
            sim_folders = [d for d in os.listdir(self.data_filepath) if os.path.isdir(os.path.join(self.data_filepath, d)) and d.startswith('mesh_')]
            num_sim_folders = len(sim_folders)
            if self.sim_limit is not None:
                num_sim_folders = min(num_sim_folders, self.sim_limit)
                sim_folders = sim_folders[:num_sim_folders]

            print(f"Processing {num_sim_folders} {self.mode} meshes:")
            for i, vtk_directory in enumerate(sim_folders):
                print(f"\tmesh {i+1}/{num_sim_folders}: {vtk_directory}")
                full_fpath = os.path.join(self.data_filepath, vtk_directory)
                mesh_name = os.path.basename(vtk_directory)

                if mesh_name in output:
                    print(f"\t\tMesh '{mesh_name}' already exists. Overwriting...")
                    del output[mesh_name]
                mesh_group = output.create_group(mesh_name)

                # timesteps
                ts_dirs = [d for d in os.listdir(os.path.join(self.data_filepath, vtk_directory))
                                            if os.path.isdir(os.path.join(self.data_filepath, vtk_directory, d)) and
                                            os.path.exists(os.path.join(self.data_filepath, vtk_directory, f"{d}.vtm"))]
                ts_dirs = self.numeric_sort(ts_dirs)
                num_timesteps = len(ts_dirs)
                if self.config.preproc.data_timestep_range:
                    start, end = self.config.preproc.data_timestep_range
                    assert num_timesteps >= end
                else:
                    start, end = [0, num_timesteps]
                ts_dirs = ts_dirs[start:end]
                num_timesteps = len(ts_dirs)
                print(f"\t\t{num_timesteps} timesteps found")

                # geometry
                _, first_mesh, first_timestep = ts_dirs[1].split("_")

                first_mesh_path = os.path.join(full_fpath, f"mesh_{first_mesh}_{first_timestep}.vtm")
                data = pv.read(first_mesh_path)
                mesh3D = data['internal']

                bounds = data.bounds # take 2d slice of 3d mesh
                z_mid = 0.5 * (bounds[4] + bounds[5])
                mesh = mesh3D.slice(normal='z', origin=(0, 0, z_mid))
                vertex_pos = mesh.points[:, :2]

                n_points = vertex_pos.shape[0] # combine boundary and internal points
                point_labels = np.full(n_points, "internal", dtype=object)
                tree = cKDTree(vertex_pos)
                patch_files = {
                    "inlet": "boundary/inlet.vtp",
                    "outlet": "boundary/outlet.vtp",
                    "walls": "boundary/walls.vtp",
                    "obstacle": "boundary/obstacle.vtp"
                }
                meta_json = pd.read_json(os.path.join(full_fpath, "meta.json"))
                topBotType = meta_json["boundary_conditions"]["walls"]["type"]
                for label, path in patch_files.items():
                    # print(f"Processing {label} patch")
                    fpath = os.path.join(full_fpath, ts_dirs[0], path)
                    patch = pv.read(fpath)
                    patch = patch.slice(normal='z', origin=(0, 0, z_mid))
                    patch_points = patch.points[:, :2]
                    matched_indices = tree.query(patch_points, k=1)[1]
                    point_labels[matched_indices] = label
                vertex_type = np.full(n_points, NodeType.NORMAL, dtype=np.int32)
                vertex_type[point_labels == "inlet"] = NodeType.INFLOW
                vertex_type[point_labels == "outlet"] = NodeType.OUTFLOW
                if topBotType == "noSlip":
                    vertex_type[point_labels == "walls"] = NodeType.WALL_BOUNDARY
                elif topBotType == "slip":
                    vertex_type[point_labels == "walls"] = NodeType.SLIP
                else:
                    vertex_type[point_labels == "walls"] = NodeType.NORMAL
                vertex_type[point_labels == "obstacle"] = NodeType.WALL_BOUNDARY

                cells = []
                for i in range(mesh.n_cells):
                    cell = mesh.get_cell(i)
                    cell_points = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
                    cells.append(cell_points)
                vertex_cell = np.array(cells, dtype=np.int32)

                geom = mesh_group.create_group('geom')
                geom = self.write_geometry(geom, vertex_pos, vertex_cell, vertex_type)
                vertex_edge_index = geom['vertex_edge_index'][()]
                cell_edge_index = geom['cell_edge_index'][()]
                num_cells = vertex_cell.shape[0]
                num_faces = vertex_edge_index.shape[1]
                face_pos   = geom['face_pos'][()]        # (nF, 2) centroids in xy
                face_normal = geom['face_normal'][()]    # (nF, 2) normal vectors in xy
                # print("created faces ", face_pos.shape)
                face_kdtree = cKDTree(face_pos)
                ## Boundary Data
                patch_to_face = {}                       # dict: patch_name -> 1-D array of face indices
                for label, rel_path in patch_files.items():          # inlet, outlet, walls, obstacle
                    patch_path = os.path.join(full_fpath, ts_dirs[0], rel_path)
                    patch      = pv.read(patch_path).slice(normal='z', origin=(0, 0, z_mid))
                    patch_centroids = patch.cell_centers().points[:, :2]
                    idx = face_kdtree.query(patch_centroids, k=1)[1]
                    # print(f"{label}: {idx}")
                    patch_to_face[label] = np.unique(idx)
                ## Face-Centred data
                surface_fields_dir = os.path.join(full_fpath, "surface-fields")
                first_surface_file = f"surfaceFields_{first_timestep}.vtp"
                first_surface_path = os.path.join(surface_fields_dir, first_surface_file)
                if os.path.exists(first_surface_path):
                    surface_mesh = pv.read(first_surface_path)
                    # surface_mesh_2d = surface_mesh.slice(normal='z', origin=(0, 0, z_mid))
                    z_coords = surface_mesh.points[:, 2]
                    z_min, z_max = z_coords.min(), z_coords.max()
                    z_tolerance = (z_max - z_min) * 0.01  # 1% tolerance
                    is_side_point = ~((np.abs(z_coords - z_min) < z_tolerance) |
                                        (np.abs(z_coords - z_max) < z_tolerance))
                    # print("surface faces", np.sum(is_side_point))
                    surface_centroids = surface_mesh.points[is_side_point, :2]
                    surface_to_face_mapping = face_kdtree.query(surface_centroids, k=1)[1]
                    # surface_to_face_mapping = np.unique(surface_to_face_mapping)
                    # print(surface_to_face_mapping)
                    # print(f"\t\tSurface fields: {surface_to_face_mapping.shape}")

                ## metadata
                meta = mesh_group.create_group('meta')
                meta_json = pd.read_json(os.path.join(full_fpath, "meta.json"))
                dt = meta_json["physics"]["dt"]
                meta['dt'] = dt
                meta["Re"] = meta_json["physics"]["Re"]
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
                face.create_dataset('flux', shape=(num_timesteps, num_faces, 1),
                                   dtype='f4', chunks=(1, num_faces, 1),
                                   compression='gzip', compression_opts=4)

                ## Process each timestep
                for ts, ts_dir in enumerate(ts_dirs):
                    print(f"\t\ttimestep {ts+1}/{num_timesteps}", end='\r', flush=True)
                    # Read the VTM file for this timestep
                    timestep_path = f"{full_fpath}/{ts_dir}.vtm"
                    timestep_data = pv.read(timestep_path)
                    mesh3D = timestep_data['internal']
                    mesh = mesh3D.slice(normal='z', origin=(0, 0, z_mid))

                    cell_velocity = mesh.cell_data['U'][:, :2]  # Keep only x,y components
                    cell_pressure = mesh.cell_data['p'].reshape(-1, 1)

                    cell_pos = geom['cell_pos'][()]
                    face_pos = geom['face_pos'][()]
                    face_velocity = custom_geom.cell_to_face(cell_velocity, cell_edge_index, face_pos, cell_pos)
                    face_pressure = custom_geom.cell_to_face(cell_pressure, cell_edge_index, face_pos, cell_pos)

                    for patch_name, face_idx in patch_to_face.items():
                        bc_path = os.path.join(full_fpath, ts_dir, f"boundary/{patch_name}.vtp")
                        bc_mesh = pv.read(bc_path).slice(normal='z', origin=(0, 0, z_mid))

                        # Initialize with defaults
                        U_patch = None
                        p_patch = None

                        # Check cell data first
                        if 'U' in bc_mesh.cell_data.keys():
                            U_patch = bc_mesh.cell_data['U'][:, :2]
                        if 'p' in bc_mesh.cell_data.keys():
                            p_patch = bc_mesh.cell_data['p'].reshape(-1, 1) # these are mostly trash

                        # CORRECT Zero-grad B.Cs
                        if patch_name == "outlet": # force zero grad manually #ERR
                            U_patch = cell_velocity[cell_edge_index[0, face_idx]]
                        else:
                            p_patch = cell_pressure[cell_edge_index[0, face_idx]]

                        # Handle missing velocity data
                        if U_patch is None or U_patch.size == 0:
                            print(f"Warning: {patch_name} has no velocity data. Using BC type instead.")
                            bc_type = meta_json["boundary_conditions"][patch_name]["type"]
                            if bc_type == 'noSlip':
                                U_patch = np.zeros((face_idx.size, 2))
                            else:
                                U_patch = np.zeros((face_idx.size, 2))  # Default to zero

                        # Handle missing pressure data
                        if p_patch is None or p_patch.size == 0:
                            print(f"Warning: {patch_name} has no pressure data.")
                            # You might want to extrapolate from internal field or use a specific BC
                            p_patch = np.zeros((face_idx.size, 1))  # Or use face_pressure[face_idx]

                        # Ensure sizes match
                        if U_patch.shape[0] != face_idx.size:
                            print(f"Warning: Size mismatch for {patch_name} velocity: {U_patch.shape[0]} vs {face_idx.size}")
                        if p_patch.shape[0] != face_idx.size:
                            print(f"Warning: Size mismatch for {patch_name} pressure: {p_patch.shape[0]} vs {face_idx.size}")

                        face_velocity[face_idx] = U_patch
                        face_pressure[face_idx] = p_patch

                    cell['velocity'][ts] = cell_velocity.astype('f4')
                    cell['pressure'][ts] = cell_pressure.astype('f4')
                    face['velocity'][ts] = face_velocity.astype('f4')
                    face['pressure'][ts] = face_pressure.astype('f4')

                    if os.path.exists(surface_fields_dir) and ts >= 0:  # Skip first timestep (no surface field)
                        # Extract timestep number from mesh directory name
                        mesh_ts_num = int(ts_dir.split('_')[-1])
                        surface_file = f"surfaceFields_{mesh_ts_num}.vtp"
                        surface_path = os.path.join(surface_fields_dir, surface_file)

                        if os.path.exists(surface_path):
                            # Read surface mesh
                            surface_mesh = pv.read(surface_path)
                            z_coords = surface_mesh.points[:, 2]
                            z_min, z_max = z_coords.min(), z_coords.max()
                            z_tolerance = (z_max - z_min) * 0.01  # 1% tolerance
                            is_side_point = ~((np.abs(z_coords - z_min) < z_tolerance) |
                                             (np.abs(z_coords - z_max) < z_tolerance))
                            # List fields in surface_mesh_2d for debugging
                            # Extract phi values
                            if 'phi' in surface_mesh.point_data:
                                phi_surface = surface_mesh.point_data['phi'][is_side_point, :2]
                                face_phi = np.zeros((num_faces, 1), dtype='f4')
                                face_phi[surface_to_face_mapping] = np.mean(phi_surface, axis=1, keepdims=True)
                                # Ensure sign consistency between phi and face velocity dot product
                                vel_dot_product = np.sum(face_normal * face_velocity, axis=1, keepdims=True)
                                phi_sign = np.sign(face_phi.flatten())
                                vel_sign = np.sign(vel_dot_product.flatten())
                                sign_mismatch = phi_sign != vel_sign
                                face_phi[sign_mismatch] *= -1

                                face['flux'][ts] = face_phi
                            else:
                                print(f"\n\t\tWarning: 'phi' not found in {surface_file}")
                                face['flux'][ts] = np.zeros((num_faces, 1), dtype='f4')
                    else:
                        # First timestep or missing surface field - use zeros
                        face['flux'][ts] = np.zeros((num_faces, 1), dtype='f4')
                print('\n')

        return

    def set_noise_std(self, stats):
        if not self.config.training.noise_std and self.config.training.noise_std != 0.0:
            mean_vel = stats['cell_velocity_x']['mean']
            self.config.training.noise_std = self.config.training.noise_std_norm * mean_vel

        print("Noise std set to:", self.config.training.noise_std)
