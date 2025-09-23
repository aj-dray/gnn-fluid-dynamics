# === LIBRARIES ===


import torch
from torch_geometric.nn import global_mean_pool, global_add_pool


# === MODULES ===


from utils.maths import chain_dot_product


# === FUNCTIONS ===


def MSE_per_element(output, target, mask, batch=None):
    """
    Compute mean squared error per element with optional masking.
    
    Args:
        output: Model predictions
        target: Ground truth values
        mask: Optional mask for selecting elements
        batch: Batch indices (unused in this function)
        
    Returns:
        torch.Tensor: Mean squared error
    """
    if mask is not None:
        output = output[mask]
        target = target[mask]
    return torch.mean((output - target) ** 2)


def MSE_per_graph(model_output, target, batch):
    """
    Compute mean squared error per graph in batch.
    
    Args:
        model_output: Model predictions
        target: Ground truth values
        batch: Batch assignment for each node
        
    Returns:
        torch.Tensor: MSE for each graph in batch
    """
    # Compute squared error for all dimensions
    mse = (model_output - target) ** 2
    node_mse = mse.mean(dim=1)
    graph_mse = global_mean_pool(node_mse, batch)
    return graph_mse


def MSE_per_element_torch(output, target, mask, batch=None):
    func = torch.nn.MSELoss(reduction="mean")
    if mask is not None:
        output = output[mask]
        target = target[mask]
    return func(output, target)

def MSE_per_batch_torch(output, target, mask, batch=None):
    func = torch.nn.MSELoss(reduction="sum")
    if mask is not None:
        output = output[mask]
        target = target[mask]
    return func(output, target)


def RelMSE_per_graph(prediction,  target, batch):
    diff = prediction - target
    if diff.ndim > 1 and diff.shape[-1] > 1:
        diff_sq = torch.sum(diff**2, dim=-1)
        target_sq = torch.sum(target**2, dim=-1)
    else:
        diff_sq = diff**2
        target_sq = target**2
        # Flatten if tensor is 2D with single feature dimension
        if diff_sq.ndim > 1:
            diff_sq = diff_sq.squeeze(-1)
            target_sq = target_sq.squeeze(-1)
    
    # Sum squared differences and target values per graph
    ssum_diff = global_add_pool(diff_sq, batch)
    ssum_gt = global_add_pool(target_sq, batch)
    
    # Compute RelMSE per graph
    RelMSE = ssum_diff / ssum_gt
    return RelMSE