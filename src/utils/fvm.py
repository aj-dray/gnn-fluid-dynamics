from utils.maths import *


def divergence_from_face_flux(face_flux, face_face):
    continuity_tensor = (
        face_flux[face_face[0]]
        + face_flux[face_face[1]]
        + face_flux[face_face[2]]
    )
    return continuity_tensor


def divergence_from_cell_flux(cell_flux):
    continuity_tensor = (
        cell_flux[:, 0]
        + cell_flux[:, 1]
        + cell_flux[:, 2]
    )
    return continuity_tensor.unsqueeze(-1)


def calc_flux_from_uf(face_velocity, face_normal, face_area):
    return (chain_dot_product(face_velocity, face_normal) * face_area).reshape(-1, 1)


def divergence_from_uf(face_velocity, cell_normal, face_area, face_face):
    e0 = face_area[face_face[0]]
    e1 = face_area[face_face[1]]
    e2 = face_area[face_face[2]]

    unv = cell_normal

    F0 = chain_flux_dot_product(face_velocity[face_face[0]], unv[:, 0, :]) * e0
    F1 = chain_flux_dot_product(face_velocity[face_face[1]], unv[:, 1, :]) * e1
    F2 = chain_flux_dot_product(face_velocity[face_face[2]], unv[:, 2, :]) * e2

    return F0 + F1 + F2


def divergence_from_uc(cell_velocity, weights, neighbours, cell_volume):
    cell_velocity_x = cell_velocity[:, 0]
    cell_velocity_y = cell_velocity[:, 1]
    neighbor_x = cell_velocity_x[neighbours]  # [n_cells, k_neighbors]
    neighbor_y = cell_velocity_y[neighbours]  # [n_cells, k_neighbors]
    diff_x = neighbor_x - cell_velocity_x[:, None]
    diff_y = neighbor_y - cell_velocity_y[:, None]
    # diff_x = neighbor_x
    # diff_y = neighbor_y
    gradient_x = torch.sum(weights[:, :, 0] * diff_x, dim=1)
    gradient_y = torch.sum(weights[:, :, 1] * diff_y, dim=1)

    return (gradient_x + gradient_y).unsqueeze(-1) * cell_volume


def convert_cell_flux_to_face_flux(
    cell_flux: torch.Tensor, 
    edge_index: torch.Tensor, 
    face_index: torch.Tensor
) -> torch.Tensor:
    """
    Convert cell flux to face flux using owner cells.
    
    Args:
        cell_flux: (num_cells, 3) - flux for each local face of each cell
        edge_index: (2, num_faces) - [owner_cells, neighbor_cells]
        face_index: (3, num_cells) - global face IDs for each cell's 3 faces
    """
    num_faces = edge_index.shape[1]
    device = cell_flux.device
    
    owner_cells = edge_index[0]  # (num_faces,)
    
    # Get face indices for each owner cell: (3, num_faces)
    owner_cell_faces = face_index[:, owner_cells]
    
    # Create face IDs and reshape for proper broadcasting
    face_ids = torch.arange(num_faces, device=device).unsqueeze(0)  # (1, num_faces)
    
    # Find which local face position corresponds to each global face
    # mask[i, j] = True if local face i of owner cell j corresponds to global face j
    mask = (owner_cell_faces == face_ids)  # (3, num_faces)
    
    # Verify each face appears exactly once in its owner cell
    matches_per_face = mask.sum(dim=0)
    if not torch.all(matches_per_face == 1):
        raise ValueError("Each face must appear exactly once in its owner cell's face list")
    
    # Get local face positions
    local_face_pos = torch.argmax(mask.int(), dim=0)  # (num_faces,)
    
    # Extract flux values
    face_flux = cell_flux[owner_cells, local_face_pos]
    
    return face_flux.unsqueeze(-1)  # (num_faces, 1)

def face_flux_to_cell_flux_vectorized(face_flux: torch.Tensor,
                                      face_face: torch.Tensor,
                                      cell_adjacency: torch.Tensor) -> torch.Tensor:
    """
    Convert face flux (owner oriented) to per-cell local face flux with signs.
    Does NOT flip sign for boundary faces where owner == neighbor (or neighbor == -1).
    
    Args:
        face_flux: (num_faces, 1) or (num_faces,) tensor
        face_face: (3, num_cells) tensor mapping each cell's local face slot -> global face id
        cell_adjacency: (2, num_faces) tensor [owners, neighbors] (neighbor can be -1 or == owner for boundary)
    Returns:
        cell_flux: (num_cells, 3) tensor; flux for each cell's 3 faces (signed, + outward)
    """
    if not (isinstance(face_flux, torch.Tensor) and
            isinstance(face_face, torch.Tensor) and
            isinstance(cell_adjacency, torch.Tensor)):
        raise TypeError("All inputs must be torch.Tensors")

    device = face_flux.device
    face_face = face_face.to(device)
    cell_adjacency = cell_adjacency.to(device)

    if cell_adjacency.shape[0] != 2:
        raise ValueError("cell_adjacency must have shape (2, num_faces)")

    num_cells = face_face.shape[1]

    # Flatten mapping: cell -> local face slot -> global face id
    face_indices = face_face.t().reshape(-1).long()  # (num_cells * 3,)
    cell_indices = torch.arange(num_cells, device=device).repeat_interleave(3)

    owners = cell_adjacency[0, face_indices]
    neighbors = cell_adjacency[1, face_indices]

    # Boundary detection
    boundary_same = owners == neighbors            # owner == neighbor (boundary style 1)
    boundary_neg1 = neighbors == -1                # neighbor == -1 (boundary style 2)
    interior_mask = ~(boundary_same | boundary_neg1)

    # Initialize signs to 0
    signs = torch.zeros_like(cell_indices)

    # Owner contribution always +1
    owner_mask = (cell_indices == owners)
    signs = torch.where(owner_mask, torch.ones_like(signs), signs)

    # Neighbor contribution only for true interior faces (owners != neighbors and neighbor != -1)
    neighbor_mask = interior_mask & (cell_indices == neighbors)
    signs = torch.where(neighbor_mask, -torch.ones_like(signs), signs)

    # Validate unresolved (should only be zeros for cells that do not own/neighbor that face)
    unresolved = (signs == 0) & interior_mask
    if torch.any(unresolved):
        raise ValueError("Inconsistent cell-face connectivity detected (unresolved interior faces).")

    # Gather flux
    face_flux_flat = face_flux.view(-1)
    flux_values = face_flux_flat[face_indices] * signs  # (num_cells * 3,)

    return flux_values.view(num_cells, 3).unsqueeze(-1)  # (num_cells, 3, 1)

def face_flux_to_cell_flux_dummy(face_flux: torch.Tensor,
                                      face_face: torch.Tensor,
                                      cell_adjacency: torch.Tensor) -> torch.Tensor:
    """
    Dummy version that assigns +1 flux on every owner cell face and -1 on the
    corresponding neighbor cell face (for interior faces). Boundary faces
    (neighbor == -1 or neighbor == owner) get only +1 on the owner side.
    This yields equal and opposite contributions for opposite directions.
    """
    if not (isinstance(face_flux, torch.Tensor) and
            isinstance(face_face, torch.Tensor) and
            isinstance(cell_adjacency, torch.Tensor)):
        raise TypeError("All inputs must be torch.Tensors")

    device = face_face.device
    owners = cell_adjacency[0].to(device)
    neighbors = cell_adjacency[1].to(device)
    num_faces = owners.shape[0]
    num_cells = face_face.shape[1]

    # Prepare output
    out = torch.zeros(num_cells, 3, device=device, dtype=face_flux.dtype)

    face_ids = torch.arange(num_faces, device=device)

    # Owner local face slot
    owner_faces = face_face[:, owners]                     # (3, num_faces)
    owner_mask = owner_faces == face_ids.unsqueeze(0)      # (3, num_faces)
    if torch.any(owner_mask.sum(dim=0) != 1):
        raise ValueError("Each face must appear exactly once in its owner cell.")
    owner_slot = torch.argmax(owner_mask.int(), dim=0)     # (num_faces,)

    out[owners, owner_slot] = 1.0

    # Neighbor (interior) local face slot: set to -1
    interior_mask = (neighbors != -1) & (neighbors != owners)
    if interior_mask.any():
        interior_neighbors = neighbors[interior_mask]
        interior_face_ids = face_ids[interior_mask]

        neighbor_faces = face_face[:, interior_neighbors]                  # (3, n_int)
        neighbor_mask = neighbor_faces == interior_face_ids.unsqueeze(0)   # (3, n_int)
        if torch.any(neighbor_mask.sum(dim=0) != 1):
            raise ValueError("Each interior face must appear exactly once in its neighbor cell.")
        neighbor_slot = torch.argmax(neighbor_mask.int(), dim=0)           # (n_int,)

        out[interior_neighbors, neighbor_slot] = -1.0

    return out.unsqueeze(-1)
