import torch

def random_edge_flip(cell_edge_index):
    rand = torch.randint(0, 2, (cell_edge_index.shape[1],), dtype=torch.bool)
    cell_edge_index[0, rand], cell_edge_index[1, rand] = \
        cell_edge_index[1, rand], cell_edge_index[0, rand]
    return cell_edge_index, rand

def calc_face_velocity_change(cell_velocity, cell_edge_index):
    face_velocity_change = cell_velocity[cell_edge_index[0]] - cell_velocity[cell_edge_index[1]]
    return face_velocity_change

def calc_cell_edge_vector(cell_pos, cell_edge_index):
    return cell_pos[cell_edge_index[0]] - cell_pos[cell_edge_index[1]]

def calc_face_type_one_hot(types, num_classes):
    return torch.nn.functional.one_hot(types.squeeze(-1), num_classes=num_classes)

def add_noise(tensor, std):
    noise = torch.normal(mean=0.0, std=std, size=tensor.shape)
    tensor += noise
    return tensor

def clean_graphs(graphs):
    c_graph, f_graph, v_graph = graphs

    del c_graph.velocity
    del c_graph.pressure

    del f_graph.velocity
    del f_graph.pressure
    del f_graph.flux

    return [c_graph, f_graph, v_graph]
