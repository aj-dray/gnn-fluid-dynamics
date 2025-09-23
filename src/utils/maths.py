import torch
import numpy as np
import os
from utils.geometry import calc_nearest_neighbours
import h5py
from tqdm import tqdm

def chain_dot_product(a, b, keepdim=True):
    return torch.sum(a * b, dim=-1, keepdim=keepdim)


def chain_flux_dot_product(a, b):
    if a.dim() < 2 or a.size(1) % 2 != 0:
        raise ValueError("a must have an even number of columns")
    if b.dim() < 2 or b.size(1) != 2:
        raise ValueError("b must have exactly 2 columns")
    parts = []
    for i in range(0, a.size(1), 2):
        parts.append(chain_dot_product(a[:, i:i+2], b))
    return torch.cat(parts, dim=-1)


class MovingLeastSquaresWeights:
    def __init__(self, config, loc='cell'):
        self.config = config
        self.loc = loc  # Store location type ('cell' or 'face')
        if self.loc == 'cell':
            self.poly_order = self.config.model.cell_grad_weights_order
        elif self.loc == 'face':
            self.poly_order = self.config.model.face_grad_weights_order
        else:
            raise ValueError(f"Unknown loc type: {self.loc}")

    def add_weights_to_dataset(self, dataset, recalculate=True):
        dataset._initialise()
        f = dataset.file_handler
        if not recalculate and 'meta' in f:
            meta = f['meta']
            if f'{self.loc}_grad_weights_orders' in meta:
                existing_poly_orders = meta[f'{self.loc}_grad_weights_orders'][()]
                if self.poly_order in existing_poly_orders:
                    print(f"\tWeights exist for {self.config.dataset.name} of poly order {self.poly_order}.")
                    return
        f.close()
        # Else precompute and save in fpath
        print(f"Computing weights for {self.config.dataset.name} of poly order {self.poly_order}.")
        self._precompute(dataset, self.poly_order, dataset.data_filepath, loc=self.loc)

    def _precompute(self, dataset, poly_order, fpath, loc='cell'):
        """
        Precompute MLS weights and neighbor indices for all meshes in dataset.
        Works with pytorch tensors.
        """
        dataset._close()
        dataset._initialise(mode='a')
        f = dataset.file_handler
        num_terms = ((poly_order+1)*(poly_order+2))//2 # num of base functions
        num_neighbours = 2 * num_terms # num of nearest neighbours

        # Get unique mesh IDs to avoid processing same geometry multiple times
        mesh_ids = dataset.get_sim_ids()
        unique_mesh_ids = list(set(mesh_ids))

        print(f"Processing {len(unique_mesh_ids)} unique meshes...")

        # Initialize progress bar for all meshes
        for mesh_id in tqdm(unique_mesh_ids, desc="Computing MLS weights"):
            # Get geometry for this mesh (direct access from the already open file)
            geom = f[mesh_id]["geom"]
            pos = torch.tensor(geom[f'{loc}_pos'][()], dtype=torch.float32)

            # Compute MLS weights
            neighbours, distances = calc_nearest_neighbours(pos, num_neighbours) #ERR: yet to add boundary conditions
            weights = self._compute_mls_weights(pos, neighbours, distances, poly_order, num_terms)

            # Save to h5 file
            mesh_group = f[mesh_id]

            if f'{loc}_grad_weights' not in mesh_group:
                grad_weights_group = mesh_group.create_group(f'{loc}_grad_weights')
            else:
                grad_weights_group = mesh_group[f'{loc}_grad_weights']

            poly_str = str(poly_order)
            if poly_str in grad_weights_group:
                del grad_weights_group[poly_str]  # Remove existing if present

            poly_group = grad_weights_group.create_group(poly_str)
            poly_group.create_dataset('neighbours', data=neighbours.numpy())
            poly_group.create_dataset('weights', data=weights)

        # Update metadata
        if 'meta' not in f:
            meta_group = f.create_group('meta')
        else:
            meta_group = f['meta']

        if f'{self.loc}_grad_weights_orders' not in meta_group:
            meta_group.create_dataset(f'{self.loc}_grad_weights_orders', data=[poly_order])
        else:
            existing_orders = list(meta_group[f'{self.loc}_grad_weights_orders'][()])
            if poly_order not in existing_orders:
                existing_orders.append(poly_order)
                del meta_group[f'{self.loc}_grad_weights_orders']
                meta_group.create_dataset(f'{self.loc}_grad_weights_orders', data=existing_orders)

        f.close()

    def _compute_mls_weights(self, cell_pos, cell_neighbours, distances, poly_order, num_terms):
        """Compute MLS weights for gradient computation"""
        n_cells, num_neighbours = cell_neighbours.shape

        weights = np.zeros((n_cells, num_neighbours, 2))

        for i in range(n_cells):
            center_pos = cell_pos[i]
            neighbor_pos = cell_pos[cell_neighbours[i]]
            rel_pos = neighbor_pos - center_pos
            # d_scale = torch.mean(distances[i])
            eps = 1e-10
            w_func = 1.0 / (distances[i] + eps)**2
            w_func /= torch.sum(w_func)
            # Build polynomial matrix P [k_neighbors, n_terms]
            P = self._build_polynomial_matrix(rel_pos, poly_order, d_scale=1.0)

            # Weighted least squares: A = P^T W P
            W = np.diag(w_func)
            A = P.T @ W @ P

            # Add regularization for numerical stability
            A += 1e-10 * np.eye(num_terms)

            try:
                A_inv = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                A_inv = np.linalg.pinv(A) # pseudo inverse

            # For gradient computation, we need derivatives of basis functions
            # Gradient of polynomial basis at origin (0,0)
            if poly_order >= 1:
                grad_basis = np.zeros((2, num_terms))
                grad_basis[0, 1] = 1.0  # d/dx of x term
                grad_basis[1, 2] = 1.0  # d/dy of y term

                if poly_order >= 2:
                    grad_basis[0, 3] = 0.0  # d/dx of x^2 at origin
                    grad_basis[0, 4] = 0.0  # d/dx of xy at origin
                    grad_basis[1, 4] = 0.0  # d/dy of xy at origin
                    grad_basis[1, 5] = 0.0  # d/dy of y^2 at origin

                if poly_order >= 3:
                    grad_basis[0, 6] = 0.0  # d/dx of x^3 at origin
                    grad_basis[0, 7] = 0.0  # d/dx of x^2y at origin
                    grad_basis[0, 8] = 0.0  # d/dx of xy^2 at origin
                    grad_basis[1, 7] = 0.0  # d/dy of x^2y at origin
                    grad_basis[1, 8] = 0.0  # d/dy of xy^2 at origin
                    grad_basis[1, 9] = 0.0  # d/dy of y^3 at origin

            for dim in range(2):  # x and y gradients
                weights[i, :, dim] = grad_basis[dim] @ A_inv @ P.T @ W

        return weights

    def _build_polynomial_matrix(self, rel_pos, poly_order, d_scale=1.0):
            """Build polynomial matrix for MLS"""
            n_points = rel_pos.shape[0]
            x, y = rel_pos[:, 0]/d_scale, rel_pos[:, 1]/d_scale

            if poly_order == 1:
                P = np.column_stack([
                    np.ones(n_points),  # 1
                    x,                  # x
                    y                   # y
                ])
            elif poly_order == 2:
                P = np.column_stack([
                    np.ones(n_points),  # 1
                    x,                  # x
                    y,                  # y
                    x**2,               # x^2
                    x * y,              # xy
                    y**2                # y^2
                ])
            elif poly_order == 3:
                P = np.column_stack([
                    np.ones(n_points),  # 1
                    x,                  # x
                    y,                  # y
                    x**2,               # x^2
                    x * y,              # xy
                    y**2,               # y^2
                    x**3,               # x^3
                    x**2 * y,           # x^2y
                    x * y**2,           # xy^2
                    y**3                # y^3
                ])
            elif poly_order == 4:
                P = np.column_stack([
                    np.ones(n_points),  # 1
                    x,                  # x
                    y,                  # y
                    x**2,               # x^2
                    x * y,              # xy
                    y**2,               # y^2
                    x**3,               # x^3
                    x**2 * y,           # x^2y
                    x * y**2,           # xy^2
                    y**3,               # y^3
                    x**4,               # x^4
                    x**3 * y,           # x^3y
                    x**2 * y**2,        # x^2y^2
                    x * y**3,           # xy^3
                    y**4                # y^4
                ])
            elif poly_order == 5:
                P = np.column_stack([
                    np.ones(n_points),  # 1
                    x,                  # x
                    y,                  # y
                    x**2,               # x^2
                    x * y,              # xy
                    y**2,               # y^2
                    x**3,               # x^3
                    x**2 * y,           # x^2y
                    x * y**2,           # xy^2
                    y**3,               # y^3
                    x**4,               # x^4
                    x**3 * y,           # x^3y
                    x**2 * y**2,        # x^2y^2
                    x * y**3,           # xy^3
                    y**4,               # y^4
                    x**5,               # x^5
                    x**4 * y,           # x^4y
                    x**3 * y**2,        # x^3y^2
                    x**2 * y**3,        # x^2y^3
                    x * y**4,           # xy^4
                    y**5                # y^5
                ])

            return P
