import numpy as np
import torch


# def interpolate_centroid(values, cell):
#     return values[cell].mean(axis=1)


# def interpolate_centroid(values, cell, vertex_pos, cell_centroids):
def interpolate_centroid(values, cell, vertex_pos, cell_centroids):
    """
    Interpolate vertex values to cell centroids using distance-weighted averaging.
    
    Args:
        values: Values at vertices
        cell: Cell-to-vertex connectivity
        vertex_pos: Vertex positions
        cell_centroids: Cell centroid positions
        
    Returns:
        np.ndarray: Interpolated values at cell centroids
    """
    # Get positions of the vertices for each cell
    # cell shape: (num_cells, 3) - indices of vertices for each triangular cell
    # vertex_pos shape: (num_vertices, 2) - x,y coordinates of each vertex
    cell_vertex_pos = vertex_pos[cell].astype(np.float64)  # Shape: (num_cells, 3, 2)
    centroids_expanded = cell_centroids[:, np.newaxis, :].astype(np.float64)

    # Calculate squared distances from each vertex to its cell centroid
    distances_sq = np.sum(
        (cell_vertex_pos - centroids_expanded) ** 2,
        axis=2
    ).astype(np.float64)  # Shape: (num_cells, 3)

    # Compute distance-proportional weights (farther = higher weight)
    # Add small epsilon to avoid division by zero when all distances are zero
    epsilon = np.float64(1e-15)
    total_distances = np.sum(distances_sq, axis=1, keepdims=True).astype(np.float64) + epsilon
    weights = (distances_sq / total_distances).astype(np.float64)  # Shape: (num_cells, 3)

    # Get values at vertices for each cell
    cell_values = values[cell].astype(np.float64)  # Shape: (num_cells, 3, num_features)

    # Apply weighted interpolation
    # weights: (num_cells, 3) -> (num_cells, 3, 1) for broadcasting
    interpolated = np.sum(
        weights[:, :, np.newaxis] * cell_values,
        axis=1
    ).astype(np.float64)  # Shape: (num_cells, num_features)

    return interpolated

# def interpolate_centroid(values, cell, pos=None, centroids=None):
#     cell_values = values[cell]  # Shape: (num_cells, 3, num_features)
#     interpolated = np.mean(cell_values, axis=1) # Shape: (num_cells, num_features)
#     return interpolated


def cross2D(a, b):
    """Compute the 2D cross product of two vectors."""
    return a[0] * b[1] - a[1] * b[0]


def compute_connectivity(cells, vertex_pos):
    """
    Computes cell-face and vertex-edge connectivity for a triangular mesh.
    Fixed to exactly match triangles_to_faces() ordering logic.

    Parameters
    ----------
    cells : numpy.ndarray
        An integer array of shape (num_cells, 3) defining
        the three vertex indices for each triangular cell.
    vertex_pos : numpy.ndarray
        A float array of shape (num_vertices, 2) specifying the (x, y)
        coordinates for each vertex.

    Returns
    -------
    face_index : numpy.ndarray
        An integer array of shape (3, num_cells). For each cell, it lists the
        three global indices of the faces that form its boundary.
    cell_edge_index : numpy.ndarray
        An integer array of shape (2, num_edges). This represents the cell
        adjacency graph. For each edge, it contains [sender_cell, receiver_cell].
        For boundary edges, this is a self-loop [cell, cell].
    edge_index : numpy.ndarray
        An integer array of shape (2, num_edges). This represents the vertex
        graph. For each edge, it contains [vertex_1, vertex_2].
    """

    # Convert cells to torch tensor
    faces = torch.from_numpy(cells).to(torch.long)
    num_cells = cells.shape[0]

    # EXACT REPLICATION OF triangles_to_faces() LOGIC
    # collect edges from triangles - same as triangles_to_faces
    edges = torch.cat(
        (
            faces[:, 0:2],
            faces[:, 1:3],
            torch.stack((faces[:, 2], faces[:, 0]), dim=1),
        ),
        dim=0,
    )

    # sort & pack edges as single tf.int64 - same as triangles_to_faces
    receivers, _ = torch.min(edges, dim=1)
    senders, _ = torch.max(edges, dim=1)
    packed_edges = torch.stack((senders, receivers), dim=1)
    unique_edges = torch.unique(
        packed_edges, return_inverse=False, return_counts=False, dim=0
    )
    senders, receivers = torch.unbind(unique_edges, dim=1)
    senders = senders.to(torch.long)
    receivers = receivers.to(torch.long)

    # This gives us edge_index (vertex-to-vertex connectivity)
    edge_index = torch.stack((senders, receivers), dim=1).T.numpy().astype(np.long)

    # Now build face_index using the SAME logic as Approach A
    # Build face dictionary for lookup
    face_list = edge_index.T  # Convert to (num_faces, 2)
    face_index_dict = {}
    for i in range(face_list.shape[0]):
        face_index_dict[str(face_list[i])] = i

    # Extract nodes_of_cell using SAME logic as triangles_to_faces
    # packed_edges contains ALL edges (including duplicates) in the order they were created
    nodes_of_cell = torch.stack(torch.chunk(packed_edges, 3, 0), dim=1).numpy()

    # Build face_index (cells_face) - same as Approach A
    face_index = np.zeros((3, num_cells), dtype=np.long)
    for i in range(nodes_of_cell.shape[0]):
        for j in range(3):
            edge_key = str(nodes_of_cell[i][j])
            face_index[j, i] = face_index_dict[edge_key]

    # Build cell_edge_index using SAME logic as Approach A
    cell_dict = {}
    for i in range(nodes_of_cell.shape[0]):
        for j in range(3):
            edge_key = str(nodes_of_cell[i][j])
            if edge_key in cell_dict:
                cell_dict[edge_key] = [cell_dict[edge_key][0], np.asarray(i, dtype=np.long)]
            else:
                cell_dict[edge_key] = [np.asarray(i, dtype=np.long)]

    # Build neighbour_cell connectivity
    num_faces = edge_index.shape[1]
    cell_edge_index = np.zeros((2, num_faces), dtype=np.long)

    for i in range(num_faces):
        face_str = str(face_list[i])
        cell_indices = cell_dict[face_str]
        if len(cell_indices) > 1:
            cell_edge_index[:, i] = [cell_indices[0], cell_indices[1]]
        else:
            # Boundary edge: self-loop
            cell_edge_index[:, i] = [cell_indices[0], cell_indices[0]]

    centroids = vertex_pos[cells].mean(axis=1)
    centroid_tensor = torch.from_numpy(centroids)
    neighbour_cell_tensor = torch.from_numpy(cell_edge_index.T)

    # Use the same reorder_face function as in Approach A
    neighbour_cell_reordered = reorder_face(centroid_tensor, neighbour_cell_tensor)
    cell_edge_index = neighbour_cell_reordered.T.numpy().astype(np.int64)

    return face_index, cell_edge_index, edge_index


def reorder_face(mesh_pos, edges):

    senders = edges[:, 0]
    receivers = edges[:, 1]

    edge_vec = torch.index_select(mesh_pos, 0, senders) - torch.index_select(
        mesh_pos, 0, receivers
    )
    e_x = torch.cat(
        (torch.ones(edge_vec.shape[0], 1), (torch.zeros(edge_vec.shape[0], 1))), dim=1
    )

    edge_vec_dot_ex = edge_vec[:, 0] * e_x[:, 0] + edge_vec[:, 1] * e_x[:, 1]

    edge_op = torch.logical_or(
        edge_vec_dot_ex > 0, torch.full(edge_vec_dot_ex.shape, False)
    )
    edge_op = torch.stack((edge_op, edge_op), dim=-1)

    edge_op_1 = torch.logical_and(edge_vec[:, 0] == 0, edge_vec[:, 1] > 0)
    edge_op_1 = torch.stack((edge_op_1, edge_op_1), dim=-1)

    unique_edges = torch.stack((senders, receivers), dim=1)
    inverse_unique_edges = torch.stack((receivers, senders), dim=1)

    edge_with_bias = torch.where(
        ((edge_op) | (edge_op_1)), unique_edges, inverse_unique_edges
    )

    return edge_with_bias


def compute_cell_normal(
    cell_centroids, cells_face_transposed, all_face_normals, all_face_centers
):
    """
    Computes outward-pointing normals using the original loop structure.

    This version corrects the logic to use each cell's own centroid for orientation,
    while preserving the loop and data assembly pattern of the original code to
    avoid any potential ordering differences.

    Args:
        cell_centroids (torch.Tensor): Centroid positions of each cell.
                                       Shape: (num_cells, 2).
        cells_face_transposed (torch.Tensor): Transposed indices of faces for each cell.
                                              This must have the shape (3, num_cells).
        all_face_normals (torch.Tensor): Pre-computed unit normals for ALL faces.
                                         Shape: (total_num_faces, 2).
        all_face_centers (torch.Tensor): Center positions of ALL faces.
                                         Shape: (total_num_faces, 2).

    Returns:
        torch.Tensor: Correctly oriented unit normal vectors for each face of each cell.
                      Shape: (num_cells, 3, 2).
    """
    # Ensure inputs are tensors for robustness
    cell_centroids = torch.as_tensor(cell_centroids, dtype=torch.float64)
    cells_face_transposed = torch.as_tensor(cells_face_transposed, dtype=torch.long)
    all_face_normals = torch.as_tensor(all_face_normals, dtype=torch.float64)
    all_face_centers = torch.as_tensor(all_face_centers, dtype=torch.float64)

    oriented_normals_set = []

    # Loop over the 3 faces of the cells, matching the original structure
    for i in range(3):
        # Get the indices for the i-th face of all cells
        face_indices = cells_face_transposed[i]

        # Gather the corresponding normal vectors and center positions
        face_uv = torch.index_select(all_face_normals, 0, face_indices)
        face_centers = torch.index_select(all_face_centers, 0, face_indices)

        # CORRECTED LOGIC: Use the full array of cell_centroids.
        # Broadcasting ensures each face_center is subtracted from its own cell's centroid.
        vec_face_to_centroid = cell_centroids - face_centers

        # Calculate the dot product
        dot_products = (face_uv[:, 0] * vec_face_to_centroid[:, 0] +
                        face_uv[:, 1] * vec_face_to_centroid[:, 1])

        # Create a boolean mask where the normal is pointing inward
        inward_mask = dot_products > 0

        # Stack the mask to apply it to both components of the normal vectors
        inward_mask_stacked = torch.stack((inward_mask, inward_mask), dim=-1)

        # Use torch.where to flip only the inward-pointing normals
        oriented_uv = torch.where(inward_mask_stacked, face_uv * (-1.0), face_uv)

        # Append the result, adding a dimension for the upcoming concatenation
        oriented_normals_set.append(oriented_uv.unsqueeze(1))

    # Concatenate the results along dimension 1 to get the final (num_cells, 3, 2) tensor
    unv = torch.cat(oriented_normals_set, dim=1)
    return unv


def compute_normal(v_graph, edge_index):
    v0 = v_graph.pos[edge_index[0]]
    v1 = v_graph.pos[edge_index[1]]
    edge_vec = v1 - v0                             # (E, 2)
    # rotate 90° and normalise
    normal = torch.stack((-edge_vec[:, 1], edge_vec[:, 0]), dim=1)  # (E, 2)
    return torch.nn.functional.normalize(normal, dim=1)


def compute_centroid(vertex_pos, vertex_cells):
    centroid = torch.zeros(vertex_pos.shape[0], 2, dtype=torch.float64)
    for i, cells in enumerate(vertex_cells):
        centroid[i] = torch.mean(vertex_pos[cells], dim=0)
    return centroid


def compute_cell_volume(v_pos, cells):
    if cells.shape[1] == 3:
        v0 = v_pos[cells[:, 0]]
        v1 = v_pos[cells[:, 1]]
        v2 = v_pos[cells[:, 2]]
        area = 0.5 * np.abs((v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1]))
    else:
        n_vertices = cells.shape[1]
        area = np.zeros(cells.shape[0])

        for i in range(cells.shape[0]):
            vertices = v_pos[cells[i]]
            # Shoelace formula
            polygon_area = 0.0
            for j in range(n_vertices):
                k = (j + 1) % n_vertices
                polygon_area += vertices[j, 0] * vertices[k, 1]
                polygon_area -= vertices[k, 0] * vertices[j, 1]
            area[i] = abs(polygon_area) / 2.0
    return area


def create_vertex_edge_index(vertex_cell):
    edges = np.concatenate([
        vertex_cell[:, [0, 1]],
        vertex_cell[:, [1, 2]],
        vertex_cell[:, [2, 0]]
    ], axis=0)  # shape (3*N_cells, 2)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    edge_index = edges.T
    return edge_index


def correct_normals(cell_pos, cell_edge_index, face_normal, face_pos):
    owners = cell_edge_index[0]
    cell_to_face_vec = face_pos - cell_pos[owners]
    flip_mask = np.sum(cell_to_face_vec * face_normal, axis=1) < 0
    face_normal[flip_mask] *= -1

    return face_normal


def classify_cells(face_index, face_types, class_types):
    """
    Classify cells based on the types of their faces.
    Each cell is assigned a type based on the combination of face types it contains.
    Args:
        face_index: Array of shape (3, n_cells) containing face indices for each cell
        face_types: Array of face types (n_faces,) - should be 1D, not (n_faces, 1)
        class_types: Object/enum containing classification constants
    Returns:
        Array of shape (n_cells, 1) containing cell type classifications
    """
    n_cells = face_index.shape[1]
    n_types = len(class_types) if hasattr(class_types, '__len__') else 4  # Assuming 4 types

    # Get face types for each cell - reshape face_types to 1D if needed
    face_types_1d = face_types.flatten() if face_types.ndim > 1 else face_types
    cell_face_types = face_types_1d[face_index]  # shape (3, n_cells)
    cell_face_types = cell_face_types.T          # shape (n_cells, 3)

    # Count how many faces of each type each cell owns
    cell_type_counts = np.zeros((n_cells, n_types), dtype=np.int32)
    for t in range(n_types):
        type_mask = (cell_face_types == t)
        cell_type_counts[:, t] = np.sum(type_mask, axis=1)

    # Initialize all cells as NORMAL
    cell_types = np.full(n_cells, class_types.NORMAL, dtype=np.int64)

    # Define constants for readability
    wall = class_types.WALL_BOUNDARY
    inflow = class_types.INFLOW
    outflow = class_types.OUTFLOW
    normal = class_types.NORMAL

    # Create masks for each face type count
    wall_count = cell_type_counts[:, wall]
    inflow_count = cell_type_counts[:, inflow]
    outflow_count = cell_type_counts[:, outflow]
    normal_count = cell_type_counts[:, normal]

    # Classification priority (most restrictive first):

    # 1. Wall boundary classification - any cell with wall faces
    wall_mask = wall_count > 0
    cell_types[wall_mask] = wall

    # 2. Inflow classification - cells with inflow faces but no wall faces
    inflow_mask = (inflow_count > 0) & (wall_count == 0)
    cell_types[inflow_mask] = inflow

    # 3. Outflow classification - cells with outflow faces but no wall or inflow faces
    outflow_mask = (outflow_count > 0) & (wall_count == 0) & (inflow_count == 0)
    cell_types[outflow_mask] = outflow

    # 4. Everything else remains NORMAL (cells with only normal faces or no special faces)

    return cell_types.reshape(-1, 1)


def classify_edges(edge_index, vertex_types, cell_edge_index, class_types):
    v1_types = vertex_types[edge_index[0]]
    v2_types = vertex_types[edge_index[1]]

    edge_types = np.full_like(v1_types, class_types.NORMAL)

    same_type_mask = v1_types == v2_types
    edge_types[same_type_mask & (v1_types == class_types.WALL_BOUNDARY)] = class_types.WALL_BOUNDARY
    edge_types[same_type_mask & (v1_types == class_types.INFLOW)] = class_types.INFLOW
    edge_types[same_type_mask & (v1_types == class_types.OUTFLOW)] = class_types.OUTFLOW
    edge_types[same_type_mask & (v1_types == class_types.SLIP)] = class_types.SLIP


    inflow_mask = (
        ((v1_types == class_types.WALL_BOUNDARY) & (v2_types == class_types.INFLOW)) |
        ((v1_types == class_types.INFLOW) & (v2_types == class_types.WALL_BOUNDARY)) |
        ((v1_types == class_types.SLIP) & (v2_types == class_types.INFLOW)) |
        ((v1_types == class_types.INFLOW) & (v2_types == class_types.SLIP))
    )
    edge_types[inflow_mask] = class_types.INFLOW

    outflow_mask = (
        ((v1_types == class_types.WALL_BOUNDARY) & (v2_types == class_types.OUTFLOW)) |
        ((v1_types == class_types.OUTFLOW) & (v2_types == class_types.WALL_BOUNDARY)) |
        ((v1_types == class_types.SLIP) & (v2_types == class_types.OUTFLOW)) |
        ((v1_types == class_types.OUTFLOW) & (v2_types == class_types.SLIP))
    )
    edge_types[outflow_mask] = class_types.OUTFLOW

    # boundary_mask = cell_edge_index[0] == cell_edge_index[1]
    # edge_types[~boundary_mask] = class_types.NORMAL

    # edge_types[edge_types == class_types.OUTFLOW] = class_types.NORMAL # TMP
    # edge_types[edge_types == class_types.WALL_BOUNDARY] = class_types.NORMAL

    return edge_types


def cell_to_face(cell_values, cell_edge_index, face_centre, cell_centres):
    # Distance-weighted interpolation
    n_edges = cell_edge_index.shape[1]

    # Calculate distances from cells to faces
    cell0_indices = cell_edge_index[0]
    cell1_indices = cell_edge_index[1]

    dist0 = np.linalg.norm(face_centre - cell_centres[cell0_indices], axis=1)
    dist1 = np.linalg.norm(face_centre - cell_centres[cell1_indices], axis=1)

    # Handle boundary faces (where both indices point to same cell)
    boundary_mask = cell0_indices == cell1_indices

    # Compute weights (inverse distance)
    weights = np.zeros((n_edges, 2))
    weights[:, 0] = 1.0 / (dist0 + 1e-10)  # Avoid division by zero
    weights[:, 1] = 1.0 / (dist1 + 1e-10)

    # For boundary faces, use only the weight of the adjacent cell
    weights[boundary_mask, 1] = 0

    # Normalize weights
    weights_sum = weights.sum(axis=1, keepdims=True)
    weights = weights / weights_sum

    # Apply weights
    face_values = (weights[:, 0, np.newaxis] * cell_values[cell0_indices] +
                    weights[:, 1, np.newaxis] * cell_values[cell1_indices])

    return face_values


def cell_to_face_torch(cell_values, cell_edge_index, face_centre, cell_centres):
    """PyTorch version of cell_to_face that preserves device and gradients."""
    # Distance-weighted interpolation
    n_edges = cell_edge_index.shape[1]

    # Calculate distances from cells to faces
    cell0_indices = cell_edge_index[0]
    cell1_indices = cell_edge_index[1]

    dist0 = torch.norm(face_centre - cell_centres[cell0_indices], dim=1)
    dist1 = torch.norm(face_centre - cell_centres[cell1_indices], dim=1)

    # Handle boundary faces (where both indices point to same cell)
    boundary_mask = cell0_indices == cell1_indices

    # Compute weights (inverse distance)
    weights = torch.zeros((n_edges, 2), device=cell_values.device, dtype=cell_values.dtype)
    weights[:, 0] = 1.0 / (dist0 + 1e-10)  # Avoid division by zero
    weights[:, 1] = 1.0 / (dist1 + 1e-10)

    # For boundary faces, use only the weight of the adjacent cell
    weights[boundary_mask, 1] = 0

    # Normalize weights
    weights_sum = weights.sum(dim=1, keepdim=True)
    weights = weights / weights_sum

    # Apply weights
    face_values = (weights[:, 0:1] * cell_values[cell0_indices] +
                   weights[:, 1:2] * cell_values[cell1_indices])

    return face_values

def interpolate_face_to_centroid(face_values, face_cell):
    # face_values: (num_faces, 1)
    # face_cell: (3, num_cells)
    cell_face_values = face_values[face_cell, 0]  # shape: (3, num_cells)
    cell_face_values = torch.mean(cell_face_values, axis=0, keepdim=True).T  # (num_cells, 1)
    return cell_face_values  # (num_cells, 1)

def calc_nearest_neighbours(cell_pos, n):
    """
    Find n nearest neighbors for each cell position using PyTorch
    Returns neighbors and distances excluding the cell itself
    """
    if isinstance(cell_pos, np.ndarray):
        cell_pos = torch.tensor(cell_pos, dtype=torch.float64)

    # Compute pairwise distances
    distances = torch.cdist(cell_pos, cell_pos)  # [n_cells, n_cells]

    # Get indices of n+1 nearest neighbors (including self)
    _, indices = torch.topk(distances, k=n+1, dim=1, largest=False)

    # Remove self (first column) and get distances
    neighbors = indices[:, 1:]  # [n_cells, n]
    neighbor_distances = torch.gather(distances, 1, neighbors)

    return neighbors, neighbor_distances

def calc_gradient_tensor(value, weights, neighbours):
    face_velocity_x = value[:, 0]
    face_velocity_y = value[:, 1]
    neighbour_values_x = face_velocity_x[neighbours]  # [n_cells, k_neighbors]
    neighbour_values_y = face_velocity_y[neighbours]  # [n_cells, k_neighbors]
    potential_diff_x = neighbour_values_x - face_velocity_x[:, None] # [n_cells, k_neighbors]
    potential_diff_y = neighbour_values_y - face_velocity_y[:, None] # [n_cells, k_neighbors]
    gradient_xx = torch.sum(weights[:, :, 0] * potential_diff_x, dim=1)
    gradient_xy = torch.sum(weights[:, :, 1] * potential_diff_y, dim=1)
    gradient_yx = torch.sum(weights[:, :, 0] * potential_diff_y, dim=1)
    gradient_yy = torch.sum(weights[:, :, 1] * potential_diff_x, dim=1)

    # gradient_tensor = torch.stack([
    #     torch.stack([gradient_xx, gradient_xy], dim=1),
    #     torch.stack([gradient_yx, gradient_yy], dim=1)
    # ], dim=1)
    gradient_tensor = torch.stack([gradient_xx, gradient_xy, gradient_yx, gradient_yy], dim=1)
    return gradient_tensor

def cell_flux_to_face_flux(cell_flux, face_to_cells, cell_faces):
    """
    Convert cell-local flux (num_cells, 3) to face flux (num_faces, 1)
    Args:
        cell_flux: (num_cells, 3) flux at each face of each cell (outward)
        face_to_cells: (2, num_faces) [owner, neighbor] cells for each face
        cell_faces: (3, num_cells) face indices for each cell
    """
    num_faces = face_to_cells.shape[1]
    face_flux = torch.zeros(num_faces, 1, device=cell_flux.device)

    # Flatten cell_faces and create corresponding cell indices
    global_face_indices = cell_faces.flatten()  # (num_cells * 3,)
    cell_indices = torch.arange(cell_flux.shape[0], device=cell_flux.device).repeat_interleave(3)  # (num_cells * 3,)
    local_face_indices = torch.arange(3, device=cell_flux.device).repeat(cell_flux.shape[0])  # (num_cells * 3,)

    # Get flux values for all cell-face pairs
    flux_values = cell_flux[cell_indices, local_face_indices]  # (num_cells * 3,)

    # Get owner cells for the corresponding global faces
    owner_cells = face_to_cells[0, global_face_indices]  # (num_cells * 3,)

    # Create mask for cells that are owners vs neighbors
    is_owner = (owner_cells == cell_indices)

    # Apply sign correction: owners keep positive, neighbors flip sign
    corrected_flux = torch.where(is_owner, flux_values, -flux_values)

    # Scatter the corrected flux values to the face_flux tensor
    face_flux[global_face_indices, 0] = corrected_flux

    return face_flux
