# === LIBRARIES ===


from os import stat
import torch
from torch.functional import norm
import torch.nn as nn
from torch_geometric.data import Data
from torch_scatter import scatter_add
import json
import numpy as np


# === MODULES ===


from utils.maths import chain_dot_product, chain_flux_dot_product, MovingLeastSquaresWeights
from utils.normalisation import CustomNormalizer, normalize_face_area
import utils.transforms as transforms
import utils.fvm as fvm
from models.Model import Model, build_mlp
from datasets.OpenFoam import NodeType
import utils.geometry as geometry
from models.Fvgn import FvgnA
from models.Mgn import MgnA, MgnC


# === FUNCTIONS ===


def build_mlp_antisym(config, in_size, hidden_size, out_size, norm_layer=False):
    """ Odd activation function, no bias,, antisymmetric output """
    layers = []
    # No bias in any linear layer
    layers.append(torch.nn.Linear(in_size, hidden_size, bias=False))
    layers.append(torch.nn.Tanh())  # Odd activation function
    layers.append(torch.nn.Linear(hidden_size, hidden_size, bias=False))
    layers.append(torch.nn.Tanh())  # Odd activation function
    layers.append(torch.nn.Linear(hidden_size, out_size, bias=False))
    module = torch.nn.Sequential(*layers)
    if norm_layer:
        return torch.nn.Sequential(module, torch.nn.LayerNorm(normalized_shape=out_size, elementwise_affine=False))
    return module


# === CLASSES ===


class ConservativeA(FvgnA):
    """Conservative message passing for Fvgn-like model"""
    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)

        # Encoder-Processor-Decoder
        self.encoder = self.Encoder(config, self.input_sizes, self.hidden_size) # encoder has both edge and cell
        processer_list = []
        for _ in range(config.model.mp_num):
            processer_list.append(self.GN_Block(config, self.hidden_size))
        self.processer_list = torch.nn.ModuleList(processer_list)
        self.decoder = self.Decoder(config, self.hidden_size, self.output_sizes) # inverse of encoder

    @classmethod
    def get_feature_sizes(cls, dataset):
        return ([2, 3 + len(dataset.class_types), 0], [0, 5, 0]) # ignoreing antisymmetric features

    @classmethod
    def transform_features(cls, dataset, graphs, mesh_id=None):
        cell_graph, face_graph, vertex_graph = graphs
        cell_velocity = cell_graph.velocity[:, 0]
        if dataset.noise:
            cell_velocity = transforms.add_noise(cell_velocity, std=dataset.config.training.noise_std)
        cell_graph.x = cell_velocity
        cell_graph.y = cell_graph.velocity[:, -1] - cell_velocity

        # Flip edges
        if dataset.mode == 'train':
            cell_graph.edge_index, flip_mask = transforms.random_edge_flip(cell_graph.edge_index)
            safe_flip = flip_mask & (cell_graph.edge_index[0] != cell_graph.edge_index[1]) # don't flip on boundaries
            face_graph.normal[safe_flip] *= -1

        # Use FVGN-like boundaries
        face_type = face_graph.type
        f_interior_mask = (face_type == dataset.class_types.NORMAL) | (face_type == dataset.class_types.OUTFLOW) | (face_type == dataset.class_types.SLIP) | (face_type == dataset.class_types.WALL_BOUNDARY)
        face_graph.boundary_mask = ~f_interior_mask.squeeze()

        face_velocity_change = transforms.calc_face_velocity_change(cell_velocity, cell_graph.edge_index)
        face_velocity_change[face_graph.boundary_mask] = face_graph.velocity[:, 0][face_graph.boundary_mask] # apply boundary conditions manually
        face_edge_vector  = transforms.calc_cell_edge_vector(cell_graph.pos, cell_graph.edge_index)
        face_type_one_hot = transforms.calc_face_type_one_hot(face_graph.type, len(dataset.class_types))
        # Calculate angle between face_edge_vector and face_normal
        face_edge_vector_norm = torch.nn.functional.normalize(face_edge_vector, dim=1)
        face_edge_distance = torch.norm(face_edge_vector, dim=1, keepdim=True)
        face_normal_norm = torch.nn.functional.normalize(face_graph.normal, dim=1)
        dot_product = (face_edge_vector_norm * face_normal_norm).sum(dim=1, keepdim=True)
        no_correction = torch.acos(torch.clamp(dot_product, -1.0, 1.0))  # angle in radians
        face_graph.x_symm = torch.cat([face_graph.area, no_correction, face_edge_distance, face_type_one_hot], dim=1)
        face_graph.x_asym = torch.cat([face_velocity_change, face_normal_norm], dim=1)
        face_graph.y = torch.cat([face_graph.velocity[:, -1], face_graph.pressure[:, -1]], dim=1)

        # Clean graphs - i.e. remove unneeded tensors
        graphs = transforms.clean_graphs([cell_graph, face_graph, vertex_graph])

        return graphs

    @classmethod
    def get_normalisation_map(cls):
        registry = {
            # symm
            "cell_velocity_x": (lambda graphs: graphs[0].x[:, 0:1], 'z_score'),
            "cell_velocity_y": (lambda graphs: graphs[0].x[:, 1:2], 'z_score'),
            "cell_velocity_change_x": (lambda graphs: graphs[0].y[:, 0:1], 'z_score'),
            "cell_velocity_change_y": (lambda graphs: graphs[0].y[:, 1:2], 'z_score'),
            "face_area": (lambda graphs: graphs[1].x_symm[:, 0:1], 'z_score'),
            "face_adjacent_distance" : (lambda graphs: graphs[1].x_symm[:, 2:3], 'z_score'),
            "face_velocity_x": (lambda graphs: graphs[1].y[:, 0:1], 'z_score'),
            "face_velocity_y": (lambda graphs: graphs[1].y[:, 1:2], 'z_score'),
            "face_pressure": (lambda graphs: graphs[1].y[:, 2:3], 'z_score'),
            "face_velocity_diff_char": (lambda graphs: torch.norm(graphs[1].x_asym[:, 0:2], dim=1), 'mean_scale'),
        }


        inputs = {
            # inputs
            "cell_velocity_x": (lambda graphs: graphs[0].x[:, 0:1], 'cell_velocity_x'),
            "cell_velocity_y": (lambda graphs: graphs[0].x[:, 1:2], 'cell_velocity_y'),
            "face_velocity_diff": (lambda graphs: graphs[1].x_asym[:, 0:2], 'face_velocity_diff_char'),
            "face_area": (lambda graphs: graphs[1].x_symm[:, 0:1], 'face_area'),
            "face_adjacent_distance": (lambda graphs: graphs[1].x_symm[:, 2:3], 'face_adjacent_distance'),
            # targets
            "cell_velocity_change_x": (lambda graphs: graphs[0].y[:, 0:1], 'cell_velocity_change_x'),
            "cell_velocity_change_y": (lambda graphs: graphs[0].y[:, 1:2], 'cell_velocity_change_y'),
            "face_velocity_x": (lambda graphs: graphs[1].y[:, 0:1], 'face_velocity_x'),
            "face_velocity_y": (lambda graphs: graphs[1].y[:, 1:2], 'face_velocity_y'),
            "face_pressure": (lambda graphs: graphs[1].y[:, 2:3], 'face_pressure')
        }

        outputs = {
            "cell_velocity_change_x": (lambda outputs: outputs[0][:, 0:1], 'cell_velocity_change_x'),
            "cell_velocity_change_y": (lambda outputs: outputs[0][:, 1:2], 'cell_velocity_change_y'),
            "face_velocity_x": (lambda outputs: outputs[1][:, 0:1], 'face_velocity_x'),
            "face_velocity_y": (lambda outputs: outputs[1][:, 1:2], 'face_velocity_y'),
            "face_pressure": (lambda outputs: outputs[1][:, 2:3], 'face_pressure')
        }

        return registry, inputs, outputs

    def update_features(self, output, input_graphs):
        """Update input graphs with predicted outputs for autoregressive rollout"""
        c_graph, f_graph, v_graph = input_graphs

        # Update cell velocity with predicted velocity
        c_graph.x = output["cell_velocity"].detach()

        face_velocity_change = transforms.calc_face_velocity_change(c_graph.x[:, :2], c_graph.edge_index)
        # mask = f_graph.boundary_mask
        face_type = f_graph.type
        mask = (face_type == NodeType.INFLOW) | (face_type == NodeType.WALL_BOUNDARY)
        face_velocity_change[mask] = f_graph.y[:, 0:2][mask] # apply boundary conditions manually

        f_graph.x_asym[:, 0:2] = face_velocity_change

        return [c_graph, f_graph, v_graph]

    def forward(self, graphs, mode='rollout'):
        graphs = self.normalizer.input(graphs)
        c_graph, f_graph, v_graph = graphs
        c_graph.edge_attr = f_graph.x_symm
        c_graph.edge_attr_asym = f_graph.x_asym
        c_graph_geom = c_graph.clone()

        c_graph = self.encoder(c_graph)

        for gnblock in self.processer_list:
            c_graph = gnblock(c_graph)

        edge_attr_out = self.decoder(c_graph)

        self.dt = c_graph_geom.dt
        acc_pred = self.integrator(edge_attr_out, c_graph_geom, f_graph, self.dt)
        output = [acc_pred, edge_attr_out, None]

        if mode == 'rollout':
            output = self.normalizer.output(output, inverse=True)

        return {
            'cell_velocity_change': output[0][:, 0:2],
            'face_velocity': output[1][:, :2],
            'face_pressure': output[1][:, 2:3],
        }

    class Encoder(torch.nn.Module):
        def __init__(self, config, input_sizes, hidden_size):
            super().__init__()
            self.faceA_mlp = build_mlp_antisym(config, 4, hidden_size, out_size=hidden_size)
            self.faceS_mlp = build_mlp(config, input_sizes[1], hidden_size, out_size=hidden_size)
            self.cell_mlp = build_mlp(config, input_sizes[0], hidden_size, out_size=hidden_size)

        def forward(self, cell_graph):
            face_attr_symm = self.faceS_mlp(cell_graph.edge_attr)
            face_attr_asym = self.faceA_mlp(cell_graph.edge_attr_asym)
            cell_attr = self.cell_mlp(cell_graph.x)
            return Data(x=cell_attr, edge_attr=face_attr_symm, edge_attr_asym=face_attr_asym, edge_index=cell_graph.edge_index)

    class GN_Block(torch.nn.Module):
        def __init__(self, config, hidden_size):
            super().__init__()
            self.face_block = self.Face_Block(config, hidden_size)
            self.cell_block = self.Cell_Block(config, hidden_size)

        def forward(self, c_graph):
            prev_edge_attr = c_graph.edge_attr.clone()
            prev_node_attr = c_graph.x.clone()

            c_graph = self.face_block(c_graph)
            c_graph = self.cell_block(c_graph)

            edge_attr = prev_edge_attr + c_graph.edge_attr # residual connections
            node_attr = prev_node_attr + c_graph.x

            return Data(x=node_attr, edge_attr=edge_attr, edge_index=c_graph.edge_index)

        class Face_Block(torch.nn.Module):
            def __init__(self, config, hidden_size):
                super().__init__()
                input_size = hidden_size * 2 # (edge_attr, sum of cell)
                self.face_mlp = build_mlp(config, input_size, hidden_size, out_size=hidden_size)

            def forward(self, c_graph):
                row, col = c_graph.edge_index
                aggr_features = torch.cat([c_graph.edge_attr, c_graph.x[row] + c_graph.x[col]], dim=1)        # (E, F_x)
                edge_attr = self.face_mlp(aggr_features)
                if hasattr(c_graph, 'edge_attr_asym'):
                    edge_attr = edge_attr * c_graph.edge_attr_asym
                return Data(x=c_graph.x, edge_attr=edge_attr, edge_index=c_graph.edge_index)

        class Cell_Block(torch.nn.Module):
            def __init__(self, config, hidden_size, mp_times=2):
                super().__init__()
                input_size = hidden_size * 2 # (cell_attr, sum of face features)
                self.cell_mlp = build_mlp(config, input_size, hidden_size, out_size=hidden_size)
                self.mp_times = mp_times

            def forward(self, c_graph):
                row, col = c_graph.edge_index
                undirected_edge_index = torch.cat([col, row], dim=0)

                edge_attr = c_graph.edge_attr
                undirected_edge_attr = torch.cat([edge_attr, -edge_attr], dim=0)  # duplicate for two-way edges
                aggregates_messages = scatter_add(undirected_edge_attr, undirected_edge_index, dim=0, dim_size=c_graph.num_nodes)

                collected_features = torch.cat([c_graph.x, aggregates_messages], dim=-1) #SUG: make this asymmetric too?
                cell_attr = self.cell_mlp(collected_features)

                return Data(x=cell_attr, edge_attr=edge_attr, edge_index=c_graph.edge_index)

    class Decoder(torch.nn.Module):
        def __init__(self, config, hidden_size, output_sizes):
            super().__init__()
            self.face_mlp = build_mlp(config, hidden_size, hidden_size, output_sizes[1], norm_layer=False)

        def forward(self, graph):
            return self.face_mlp(graph.edge_attr)


class ConservativeB(MgnA):
    """Conservative message passing for Mgn-like model"""
    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)

        # Encoder-Processor-Decoder
        self.encoder = ConservativeA.Encoder(config, self.input_sizes, self.hidden_size) # encoder has both edge and cell
        processer_list = []
        for _ in range(config.model.mp_num):
            processer_list.append(ConservativeA.GN_Block(config, self.hidden_size))
        self.processer_list = torch.nn.ModuleList(processer_list)

    @classmethod
    def get_feature_sizes(cls, dataset):
        return ([2, 3 + len(dataset.class_types), 0], [3, 0, 0])

    @classmethod
    def transform_features(cls, dataset, graphs, mesh_id=None):
        cell_graph, face_graph, vertex_graph = graphs
        cell_velocity = cell_graph.velocity[:, 0]
        if dataset.noise:
            cell_velocity = transforms.add_noise(cell_velocity, std=dataset.config.training.noise_std)
        cell_graph.x = cell_velocity
        cell_velocity_target = cell_graph.velocity[:, -1]
        cell_velocity_change = cell_velocity_target - cell_velocity
        cell_graph.y = torch.cat([cell_velocity_change, cell_graph.pressure[:, -1]], dim=1)

        # Flip edges
        if dataset.mode == 'train':
            cell_graph.edge_index, flip_mask = transforms.random_edge_flip(cell_graph.edge_index)
            safe_flip = flip_mask & (cell_graph.edge_index[0] != cell_graph.edge_index[1]) # don't flip on boundaries
            face_graph.normal[safe_flip] *= -1

        # Use FVGN-like boundaries
        face_type = face_graph.type
        f_interior_mask = (face_type == dataset.class_types.NORMAL) | (face_type == dataset.class_types.OUTFLOW) | (face_type == dataset.class_types.SLIP) | (face_type == dataset.class_types.WALL_BOUNDARY)
        face_graph.boundary_mask = ~f_interior_mask.squeeze()

        face_velocity_change = transforms.calc_face_velocity_change(cell_velocity, cell_graph.edge_index)
        face_velocity_change[face_graph.boundary_mask] = face_graph.velocity[:, 0][face_graph.boundary_mask] # apply boundary conditions manually
        face_edge_vector  = transforms.calc_cell_edge_vector(cell_graph.pos, cell_graph.edge_index)
        face_type_one_hot = transforms.calc_face_type_one_hot(face_graph.type, len(dataset.class_types))
        # Calculate angle between face_edge_vector and face_normal
        face_edge_vector_norm = torch.nn.functional.normalize(face_edge_vector, dim=1)
        face_edge_distance = torch.norm(face_edge_vector, dim=1, keepdim=True)
        face_normal_norm = torch.nn.functional.normalize(face_graph.normal, dim=1)
        dot_product = (face_edge_vector_norm * face_normal_norm).sum(dim=1, keepdim=True)
        no_correction = torch.acos(torch.clamp(dot_product, -1.0, 1.0))  # angle in radians
        face_graph.x_symm = torch.cat([face_graph.area, no_correction, face_edge_distance, face_type_one_hot], dim=1)
        face_graph.x_asym = torch.cat([face_velocity_change, face_normal_norm], dim=1)
        face_graph.y = face_graph.velocity[:, -1] # for boundary conditions
        # print(face_graph.x_symm.shape)

        # Clean graphs - i.e. remove unneeded tensors
        graphs = transforms.clean_graphs([cell_graph, face_graph, vertex_graph])

        return graphs

    @classmethod
    def get_normalisation_map(cls):
        registry = {
            # symm
            "cell_velocity_x": (lambda graphs: graphs[0].x[:, 0:1], 'z_score'),
            "cell_velocity_y": (lambda graphs: graphs[0].x[:, 1:2], 'z_score'),
            "cell_velocity_change_x": (lambda graphs: graphs[0].y[:, 0:1], 'z_score'),
            "cell_velocity_change_y": (lambda graphs: graphs[0].y[:, 1:2], 'z_score'),
            "cell_pressure": (lambda graphs: graphs[0].y[:, 2:3], 'z_score'),

            "face_area": (lambda graphs: graphs[1].x_symm[:, 0:1], 'z_score'),
            "face_adjacent_distance" : (lambda graphs: graphs[1].x_symm[:, 2:3], 'z_score'),
            "face_velocity_x": (lambda graphs: graphs[1].y[:, 0:1], 'z_score'),
            "face_velocity_y": (lambda graphs: graphs[1].y[:, 1:2], 'z_score'),
            # asymm
            "face_velocity_diff_char": (lambda graphs: torch.norm(graphs[1].x_asym[:, 0:2], dim=1), 'mean_scale'),
                # normal already normalised
        }

        inputs = {
            # inputs
            "cell_velocity_x": (lambda graphs: graphs[0].x[:, 0:1], 'cell_velocity_x'),
            "cell_velocity_y": (lambda graphs: graphs[0].x[:, 1:2], 'cell_velocity_y'),
            "face_velocity_diff": (lambda graphs: graphs[1].x_asym[:, 0:2], 'face_velocity_diff_char'),
            "face_area": (lambda graphs: graphs[1].x_symm[:, 0:1], 'face_area'),
            "face_adjacent_distance": (lambda graphs: graphs[1].x_symm[:, 2:3], 'face_adjacent_distance'),
            # targets
            "cell_velocity_change_x": (lambda graphs: graphs[0].y[:, 0:1], 'cell_velocity_change_x'),
            "cell_velocity_change_y": (lambda graphs: graphs[0].y[:, 1:2], 'cell_velocity_change_y'),
            "cell_pressure": (lambda graphs: graphs[0].y[:, 2:3], 'cell_pressure'),
            # bc
            "face_velocity_x": (lambda graphs: graphs[1].y[:, 0:1], 'face_velocity_x'),
            "face_velocity_y": (lambda graphs: graphs[1].y[:, 1:2], 'face_velocity_y'),
        }

        outputs = {
            "cell_velocity_change_x": (lambda outputs: outputs[0][:, 0:1], 'cell_velocity_change_x'),
            "cell_velocity_change_y": (lambda outputs: outputs[0][:, 1:2], 'cell_velocity_change_y'),
            "cell_pressure": (lambda outputs: outputs[0][:, 2:3], 'cell_pressure'),
        }

        return registry, inputs, outputs

    def update_features(self, output, input_graphs):
        """Update input graphs with predicted outputs for autoregressive rollout"""
        c_graph, f_graph, v_graph = input_graphs

        # Update cell velocity with predicted velocity
        c_graph.x = output["cell_velocity"].detach()

        face_velocity_change = transforms.calc_face_velocity_change(c_graph.x[:, :2], c_graph.edge_index)
        # mask = f_graph.boundary_mask
        face_type = f_graph.type
        mask = (face_type == NodeType.INFLOW) | (face_type == NodeType.WALL_BOUNDARY)
        face_velocity_change[mask] = f_graph.y[:, 0:2][mask] # apply boundary conditions manually

        f_graph.x_asym[:, 0:2] = face_velocity_change

        return [c_graph, f_graph, v_graph]

    def forward(self, graphs, mode='rollout'):
        graphs = self.normalizer.input(graphs)
        c_graph, f_graph, v_graph = graphs
        c_graph.edge_attr = f_graph.x_symm
        c_graph.edge_attr_asym = f_graph.x_asym
        c_graph_geom = c_graph.clone()

        c_graph = self.encoder(c_graph)

        for gnblock in self.processer_list:
            c_graph = gnblock(c_graph)

        cell_output = self.decoder(c_graph)
        output = [cell_output, None, None]

        if mode == 'rollout':
            output = self.normalizer.output(output, inverse=True)

        return {
            'cell_velocity_change': output[0][:, 0:2],
            'cell_pressure': output[0][:, 2:3],
        }

    class Decoder(torch.nn.Module):
        def __init__(self, config, hidden_size, output_sizes):
            super().__init__()
            # print(f"Decoder output sizes: {output_sizes[0]}")
            self.node_mlp = build_mlp(config, hidden_size, hidden_size, output_sizes[0], norm_layer=False)

        def forward(self, graph):
            # print(f"Decoder input shape: {graph.x.shape}")
            return self.node_mlp(graph.x)


class ConservativeD(FvgnA):
    """Conservative message passing for Fvgn-like model with enforced antisymmetry in decoder"""
    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)

        # Encoder-Processor-Decoder
        self.encoder = self.Encoder(config, self.input_sizes, self.hidden_size)
        processer_list = []
        for _ in range(config.model.mp_num):
            processer_list.append(self.GN_Block(config, self.hidden_size))
        self.processer_list = torch.nn.ModuleList(processer_list)
        self.decoder = self.Decoder(config, self.hidden_size, self.output_sizes)


    @classmethod
    def get_feature_sizes(cls, dataset):
        return ([2, 3 + len(dataset.class_types), 0], [0, 5, 0]) # ignoreing antisymmetric features

    @classmethod
    def transform_features(cls, dataset, graphs, mesh_id=None):
        cell_graph, face_graph, vertex_graph = graphs
        cell_velocity = cell_graph.velocity[:, 0]
        if dataset.noise:
            cell_velocity = transforms.add_noise(cell_velocity, std=dataset.config.training.noise_std)
        cell_graph.x = cell_velocity
        cell_graph.y = cell_graph.velocity[:, -1] - cell_velocity

        # Flip edges
        if dataset.mode == 'train':
            cell_graph.edge_index, flip_mask = transforms.random_edge_flip(cell_graph.edge_index)
            safe_flip = flip_mask & (cell_graph.edge_index[0] != cell_graph.edge_index[1]) # don't flip on boundaries
            face_graph.normal[safe_flip] *= -1

        # Use FVGN-like boundaries
        face_type = face_graph.type
        f_interior_mask = (face_type == dataset.class_types.NORMAL) | (face_type == dataset.class_types.OUTFLOW) | (face_type == dataset.class_types.SLIP) | (face_type == dataset.class_types.WALL_BOUNDARY)
        face_graph.boundary_mask = ~f_interior_mask.squeeze()

        face_velocity_change = transforms.calc_face_velocity_change(cell_velocity, cell_graph.edge_index)
        face_velocity_change[face_graph.boundary_mask] = face_graph.velocity[:, 0][face_graph.boundary_mask] # apply boundary conditions manually
        face_edge_vector  = transforms.calc_cell_edge_vector(cell_graph.pos, cell_graph.edge_index)
        face_type_one_hot = transforms.calc_face_type_one_hot(face_graph.type, len(dataset.class_types))
        # Calculate angle between face_edge_vector and face_normal
        face_edge_vector_norm = torch.nn.functional.normalize(face_edge_vector, dim=1)
        face_edge_distance = torch.norm(face_edge_vector, dim=1, keepdim=True)
        face_normal_norm = torch.nn.functional.normalize(face_graph.normal, dim=1)
        dot_product = (face_edge_vector_norm * face_normal_norm).sum(dim=1, keepdim=True)
        no_correction = torch.acos(torch.clamp(dot_product, -1.0, 1.0))  # angle in radians
        face_graph.x_symm = torch.cat([face_graph.area, no_correction, face_edge_distance, face_type_one_hot], dim=1)
        face_graph.x_asym = torch.cat([face_velocity_change, face_normal_norm], dim=1)
        face_graph.y = torch.cat([face_graph.velocity[:, -1], face_graph.pressure[:, -1]], dim=1)

        # Clean graphs - i.e. remove unneeded tensors
        graphs = transforms.clean_graphs([cell_graph, face_graph, vertex_graph])

        return graphs

    @classmethod
    def get_normalisation_map(cls):
        registry = {
            # symm
            "cell_velocity_x": (lambda graphs: graphs[0].x[:, 0:1], 'z_score'),
            "cell_velocity_y": (lambda graphs: graphs[0].x[:, 1:2], 'z_score'),
            "cell_velocity_change_x": (lambda graphs: graphs[0].y[:, 0:1], 'z_score'),
            "cell_velocity_change_y": (lambda graphs: graphs[0].y[:, 1:2], 'z_score'),
            "face_area": (lambda graphs: graphs[1].x_symm[:, 0:1], 'z_score'),
            "face_adjacent_distance" : (lambda graphs: graphs[1].x_symm[:, 2:3], 'z_score'),
            "face_velocity_x": (lambda graphs: graphs[1].y[:, 0:1], 'z_score'),
            "face_velocity_y": (lambda graphs: graphs[1].y[:, 1:2], 'z_score'),
            "face_pressure": (lambda graphs: graphs[1].y[:, 2:3], 'z_score'),
            "face_velocity_diff_char": (lambda graphs: torch.norm(graphs[1].x_asym[:, 0:2], dim=1), 'mean_scale'),
        }


        inputs = {
            # inputs
            "cell_velocity_x": (lambda graphs: graphs[0].x[:, 0:1], 'cell_velocity_x'),
            "cell_velocity_y": (lambda graphs: graphs[0].x[:, 1:2], 'cell_velocity_y'),
            "face_velocity_diff": (lambda graphs: graphs[1].x_asym[:, 0:2], 'face_velocity_diff_char'),
            "face_area": (lambda graphs: graphs[1].x_symm[:, 0:1], 'face_area'),
            "face_adjacent_distance": (lambda graphs: graphs[1].x_symm[:, 2:3], 'face_adjacent_distance'),
            # targets
            "cell_velocity_change_x": (lambda graphs: graphs[0].y[:, 0:1], 'cell_velocity_change_x'),
            "cell_velocity_change_y": (lambda graphs: graphs[0].y[:, 1:2], 'cell_velocity_change_y'),
            "face_velocity_x": (lambda graphs: graphs[1].y[:, 0:1], 'face_velocity_x'),
            "face_velocity_y": (lambda graphs: graphs[1].y[:, 1:2], 'face_velocity_y'),
            "face_pressure": (lambda graphs: graphs[1].y[:, 2:3], 'face_pressure')
        }

        outputs = {
            "cell_velocity_change_x": (lambda outputs: outputs[0][:, 0:1], 'cell_velocity_change_x'),
            "cell_velocity_change_y": (lambda outputs: outputs[0][:, 1:2], 'cell_velocity_change_y'),
            "face_velocity_x": (lambda outputs: outputs[1][:, 0:1], 'face_velocity_x'),
            "face_velocity_y": (lambda outputs: outputs[1][:, 1:2], 'face_velocity_y'),
            "face_pressure": (lambda outputs: outputs[1][:, 2:3], 'face_pressure')
        }

        return registry, inputs, outputs

    def update_features(self, output, input_graphs):
        """Update input graphs with predicted outputs for autoregressive rollout"""
        c_graph, f_graph, v_graph = input_graphs

        # Update cell velocity with predicted velocity
        c_graph.x = output["cell_velocity"].detach()

        face_velocity_change = transforms.calc_face_velocity_change(c_graph.x[:, :2], c_graph.edge_index)
        # mask = f_graph.boundary_mask
        face_type = f_graph.type
        mask = (face_type == NodeType.INFLOW) | (face_type == NodeType.WALL_BOUNDARY)
        face_velocity_change[mask] = f_graph.y[:, 0:2][mask] # apply boundary conditions manually

        f_graph.x_asym[:, 0:2] = face_velocity_change

        return [c_graph, f_graph, v_graph]


    def forward(self, graphs, mode='rollout'):
        graphs = self.normalizer.input(graphs)
        c_graph, f_graph, v_graph = graphs
        c_graph.edge_attr = f_graph.x_symm
        c_graph.edge_attr_asym = f_graph.x_asym
        c_graph_geom = c_graph.clone()

        c_graph = self.encoder(c_graph)
        for gnblock in self.processer_list:
            c_graph = gnblock(c_graph)
        edge_attr_out = self.decoder(c_graph)

        self.dt = c_graph_geom.dt
        acc_pred = self.integrator(edge_attr_out, c_graph_geom, f_graph, self.dt)
        output = [acc_pred, edge_attr_out, None]

        if mode == 'rollout':
            output = self.normalizer.output(output, inverse=True)

        return {
            'cell_velocity_change': output[0][:, 0:2],
            'face_velocity': output[1][:, :2],
            'face_pressure': output[1][:, 2:3],
        }

    class Encoder(torch.nn.Module):
        def __init__(self, config, input_sizes, hidden_size):
            super().__init__()
            self.faceA_mlp = build_mlp_antisym(config, 4, hidden_size, out_size=hidden_size)
            self.faceS_mlp = build_mlp(config, input_sizes[1], hidden_size, out_size=hidden_size)
            self.cell_mlp = build_mlp(config, input_sizes[0], hidden_size, out_size=hidden_size)

        def forward(self, cell_graph):
            face_attr_symm = self.faceS_mlp(cell_graph.edge_attr)
            face_attr_asym = self.faceA_mlp(cell_graph.edge_attr_asym)
            cell_attr = self.cell_mlp(cell_graph.x)
            return Data(x=cell_attr, edge_attr=face_attr_symm, edge_attr_asym=face_attr_asym, edge_index=cell_graph.edge_index)

    class GN_Block(torch.nn.Module):
        def __init__(self, config, hidden_size):
            super().__init__()
            self.face_block_symm = self.Face_Block_Symm(config, hidden_size)
            self.face_block_asym = self.Face_Block_Asym(config, hidden_size)
            self.cell_block = self.Cell_Block(config, hidden_size)

        def forward(self, c_graph):
            prev_edge_attr_symm = c_graph.edge_attr.clone()
            prev_edge_attr_asym = c_graph.edge_attr_asym.clone()
            prev_node_attr = c_graph.x.clone()

            edge_attr_symm = self.face_block_symm(c_graph)
            edge_attr_asym = self.face_block_asym(c_graph)

            c_graph_updated = Data(
                x=c_graph.x,
                edge_attr=edge_attr_symm,
                edge_attr_asym=edge_attr_asym,
                edge_index=c_graph.edge_index
            )
            c_graph_updated = self.cell_block(c_graph_updated)

            edge_attr_symm = prev_edge_attr_symm + c_graph_updated.edge_attr
            edge_attr_asym = prev_edge_attr_asym + c_graph_updated.edge_attr_asym
            node_attr = prev_node_attr + c_graph_updated.x

            return Data(x=node_attr, edge_attr=edge_attr_symm,
                        edge_attr_asym=edge_attr_asym, edge_index=c_graph.edge_index)

        class Face_Block_Symm(torch.nn.Module):
            def __init__(self, config, hidden_size):
                super().__init__()
                input_size = hidden_size * 2
                self.face_mlp = build_mlp(config, input_size, hidden_size, out_size=hidden_size)

            def forward(self, c_graph):
                row, col = c_graph.edge_index
                aggr_features = torch.cat([c_graph.edge_attr, c_graph.x[row] + c_graph.x[col]], dim=1)
                return self.face_mlp(aggr_features)

        class Face_Block_Asym(torch.nn.Module):
            def __init__(self, config, hidden_size):
                super().__init__()
                input_size = hidden_size * 2
                self.face_mlp = build_mlp_antisym(config, input_size, hidden_size, out_size=hidden_size)

            def forward(self, c_graph):
                row, col = c_graph.edge_index
                aggr_features = torch.cat([c_graph.edge_attr_asym, c_graph.x[row] - c_graph.x[col]], dim=1)
                return self.face_mlp(aggr_features)

        class Cell_Block(torch.nn.Module):
            def __init__(self, config, hidden_size, mp_times=2):
                super().__init__()
                input_size = hidden_size * 3
                self.cell_mlp = build_mlp(config, input_size, hidden_size, out_size=hidden_size)
                self.mp_times = mp_times

            def forward(self, c_graph):
                row, col = c_graph.edge_index
                undirected_edge_index = torch.cat([col, row], dim=0)

                undirected_edge_attr_symm = torch.cat([c_graph.edge_attr, c_graph.edge_attr], dim=0)
                undirected_edge_attr_asym = torch.cat([c_graph.edge_attr_asym, -c_graph.edge_attr_asym], dim=0)

                symm_messages = scatter_add(undirected_edge_attr_symm, undirected_edge_index, dim=0, dim_size=c_graph.num_nodes)
                asym_messages = scatter_add(undirected_edge_attr_asym, undirected_edge_index, dim=0, dim_size=c_graph.num_nodes)

                collected_features = torch.cat([c_graph.x, symm_messages, asym_messages], dim=-1)
                cell_attr = self.cell_mlp(collected_features)

                return Data(x=cell_attr, edge_attr=c_graph.edge_attr,
                            edge_attr_asym=c_graph.edge_attr_asym, edge_index=c_graph.edge_index)

    class Decoder(torch.nn.Module):
        def __init__(self, config, hidden_size, output_sizes):
            super().__init__()
            self.symm_mlp = build_mlp(config, hidden_size, hidden_size, hidden_size, norm_layer=False)
            self.asym_mlp = build_mlp_antisym(config, hidden_size, hidden_size, hidden_size, norm_layer=False)
            self.final_mlp = build_mlp_antisym(config, hidden_size, hidden_size, output_sizes[1], norm_layer=False)

        def forward(self, graph):
            symm_features = self.symm_mlp(graph.edge_attr)
            asym_features = self.asym_mlp(graph.edge_attr_asym)
            combined = symm_features + asym_features
            return self.final_mlp(combined)


class ConservativeE(FvgnA):
    """Simplest change to EPD structure - antisymmetric messages"""
    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)

        processer_list = []
        for _ in range(config.model.mp_num):
            processer_list.append(self.GN_Block(config, self.hidden_size))
        self.processer_list = torch.nn.ModuleList(processer_list)

    class GN_Block(torch.nn.Module):
        def __init__(self, config, hidden_size):
            super().__init__()
            self.face_block = self.Face_Block(config, hidden_size)
            self.cell_block = self.Cell_Block(config, hidden_size)

        def forward(self, c_graph, v_graph):
            prev_edge_attr = c_graph.edge_attr.clone()
            prev_node_attr = c_graph.x.clone()

            c_graph = self.face_block(c_graph)
            c_graph = self.cell_block(c_graph)

            edge_attr = prev_edge_attr + c_graph.edge_attr # residual connections
            node_attr = prev_node_attr + c_graph.x

            return Data(x=node_attr, edge_attr=edge_attr, edge_index=c_graph.edge_index)

        class Face_Block(torch.nn.Module):
            def __init__(self, config, hidden_size):
                super().__init__()
                input_size = hidden_size * 2 # (edge_attr, sum of cell)
                self.face_mlp = build_mlp(config, input_size, hidden_size, out_size=hidden_size)

            def forward(self, c_graph):
                row, col = c_graph.edge_index
                aggr_features = torch.cat([c_graph.edge_attr, c_graph.x[row] + c_graph.x[col]], dim=1)        # (E, F_x)
                edge_attr = self.face_mlp(aggr_features)
                return Data(x=c_graph.x, edge_attr=edge_attr, edge_index=c_graph.edge_index)

        class Cell_Block(torch.nn.Module):
            def __init__(self, config, hidden_size, mp_times=2):
                super().__init__()
                input_size = hidden_size * 2  # (cell_attr, sum of face features)
                self.cell_mlp = build_mlp(config, input_size, hidden_size, out_size=hidden_size)
                self.mp_times = mp_times

            def forward(self, c_graph):
                # Split edge attributes into symmetric and antisymmetric parts
                edge_sym, edge_asym = torch.chunk(c_graph.edge_attr, 2, dim=-1)

                row, col = c_graph.edge_index
                undirected_edge_index = torch.cat([col, row], dim=0)

                # Handle symmetric part: same value in both directions
                undirected_edge_sym = torch.cat([edge_sym, edge_sym], dim=0)
                sym_messages = scatter_add(undirected_edge_sym, undirected_edge_index,
                                        dim=0, dim_size=c_graph.num_nodes)

                # Handle antisymmetric part: negated in reverse direction
                undirected_edge_asym = torch.cat([edge_asym, -edge_asym], dim=0)
                asym_messages = scatter_add(undirected_edge_asym, undirected_edge_index,
                                            dim=0, dim_size=c_graph.num_nodes)

                # Combine both symmetric and antisymmetric aggregations
                aggregated_messages = torch.cat([sym_messages, asym_messages], dim=-1)

                # Concatenate with cell features
                collected_features = torch.cat([c_graph.x, aggregated_messages], dim=-1)
                cell_attr = self.cell_mlp(collected_features)

                return Data(x=cell_attr, edge_attr=c_graph.edge_attr, edge_index=c_graph.edge_index)

class ConservativeF(FvgnA):
    """Antisymmetric message passing for Fvgn-like model with multiple GN blocks"""
    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)

        processer_list = []
        for _ in range(config.model.mp_num):
            processer_list.append(self.GN_Block(config, self.hidden_size))
        self.processer_list = torch.nn.ModuleList(processer_list)

    # class Encoder(torch.nn.Module):
    #     def __init__(self, config, input_sizes, hidden_size):
    #         super().__init__()
    #         self.faceA_mlp = build_mlp_antisym(config, 4, hidden_size, out_size=hidden_size//2)
    #         self.faceS_mlp = build_mlp(config, input_sizes[1], hidden_size, out_size=hidden_size//2)
    #         self.cell_mlp = build_mlp(config, input_sizes[0], hidden_size, out_size=hidden_size)

    #     def forward(self, cell_graph):
    #         face_attr_symm = self.faceS_mlp(cell_graph.edge_attr)
    #         face_attr_asym = self.faceA_mlp(cell_graph.edge_attr_asym)
    #         cell_attr = self.cell_mlp(cell_graph.x)
    #         return Data(x=cell_attr, edge_attr_symm=face_attr_symm, edge_attr_asym=face_attr_asym, edge_index=cell_graph.edge_index)

    class GN_Block(torch.nn.Module):
        def __init__(self, config, hidden_size):
            super().__init__()
            self.cell_block = self.Cell_Block(config, hidden_size)
            self.face_block = self.Face_Block(config, hidden_size)

        def forward(self, c_graph, v_graph):
            prev_edge_attr = c_graph.edge_attr.clone()
            prev_node_attr = c_graph.x.clone()

            c_graph = self.cell_block(c_graph, v_graph)
            c_graph = self.face_block(c_graph)

            edge_attr = prev_edge_attr + c_graph.edge_attr
            node_attr = prev_node_attr + c_graph.x

            return Data(x=node_attr, edge_attr=edge_attr, edge_index=c_graph.edge_index)

        class Cell_Block(torch.nn.Module):
            def __init__(self, config, hidden_size):
                super().__init__()
                input_size = hidden_size * 2
                self.cell_mlp = build_mlp(config, input_size, hidden_size, out_size=hidden_size)
                # self.cell_mlp_asym = build_mlp_antisym(config, input_size, hidden_size, out_size=hidden_size//2)

            def forward(self, c_graph, v_graph):
                # edge_sym = c_graph.edge_attr_symm
                # edge_asym = c_graph.edge_attr_asym
                edge_sym, edge_asym = torch.chunk(c_graph.edge_attr, 2, dim=-1)

                # Symmetric part aggregate via vertices
                senders_node_idx, receivers_node_idx = v_graph.edge_index
                twoway_node_connections = torch.cat([senders_node_idx, receivers_node_idx], dim=0)
                twoway_edge_attr = torch.cat([edge_sym, edge_sym], dim=0)
                node_agg_received_edges = scatter_add(twoway_edge_attr, twoway_node_connections,
                                                      dim=0, dim_size=v_graph.num_nodes)
                cell_agg = (
                    torch.index_select(node_agg_received_edges, 0, v_graph.face[0]) +
                    torch.index_select(node_agg_received_edges, 0, v_graph.face[1]) +
                    torch.index_select(node_agg_received_edges, 0, v_graph.face[2])
                ) / 3.0

                row, col = c_graph.edge_index
                undirected_edge_index = torch.cat([col, row], dim=0)
                asym_messages = torch.cat([edge_asym, -edge_asym], dim=0)
                asym_agg = scatter_add(asym_messages, undirected_edge_index,
                                       dim=0, dim_size=c_graph.num_nodes)

                # Combine all features
                collected_features = torch.cat([c_graph.x, cell_agg, asym_agg], dim=-1)
                cell_attr = self.cell_mlp(collected_features)

                return Data(x=cell_attr, edge_attr=c_graph.edge_attr, edge_index=c_graph.edge_index)

        class Face_Block(torch.nn.Module):
            def __init__(self, config, hidden_size):
                super().__init__()
                input_size = hidden_size * 3  # edge + two cells
                self.face_mlp = build_mlp(config, input_size, hidden_size, out_size=hidden_size)

            def forward(self, c_graph):
                row, col = c_graph.edge_index
                aggr_features = torch.cat([c_graph.edge_attr, c_graph.x[row], c_graph.x[col]], dim=1)        # (E, F_x)
                edge_attr = self.face_mlp(aggr_features)
                return Data(x=c_graph.x, edge_attr=edge_attr, edge_index=c_graph.edge_index)


class ConservativeG(FvgnA):
    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)

        self.encoder = self.Encoder(config, self.input_sizes, self.hidden_size)
        processer_list = []
        for _ in range(config.model.mp_num):
            processer_list.append(self.GN_Block(config, self.hidden_size))
        self.processer_list = torch.nn.ModuleList(processer_list)

    class GN_Block(torch.nn.Module):
        def __init__(self, config, hidden_size):
            super().__init__()
            self.cell_block = self.Cell_Block(config, hidden_size)
            self.face_block = self.Face_Block(config, hidden_size)

        def forward(self, c_graph, v_graph):
            prev_edge_attr = c_graph.edge_attr.clone()
            prev_node_attr = c_graph.x.clone()

            c_graph = self.cell_block(c_graph, v_graph)
            c_graph = self.face_block(c_graph)

            edge_attr = prev_edge_attr + c_graph.edge_attr
            node_attr = prev_node_attr + c_graph.x

            return Data(x=node_attr, edge_attr=edge_attr, edge_index=c_graph.edge_index)

        class Cell_Block(torch.nn.Module):
            def __init__(self, config, hidden_size):
                super().__init__()
                input_size = hidden_size * 2
                self.cell_mlp = build_mlp(config, input_size, hidden_size, out_size=hidden_size)

            def forward(self, c_graph, v_graph):
                edge_sym, edge_asym = torch.chunk(c_graph.edge_attr, 2, dim=-1)

                # Symmetric part aggregate via vertices
                senders_node_idx, receivers_node_idx = v_graph.edge_index
                twoway_node_connections = torch.cat([senders_node_idx, receivers_node_idx], dim=0)
                twoway_edge_attr = torch.cat([edge_sym, edge_sym], dim=0)
                node_agg_received_edges = scatter_add(twoway_edge_attr, twoway_node_connections,
                                                      dim=0, dim_size=v_graph.num_nodes)
                cell_agg = (
                    torch.index_select(node_agg_received_edges, 0, v_graph.face[0]) +
                    torch.index_select(node_agg_received_edges, 0, v_graph.face[1]) +
                    torch.index_select(node_agg_received_edges, 0, v_graph.face[2])
                ) / 3.0

                # Asymmetric part aggregate via edges
                row, col = c_graph.edge_index
                undirected_edge_index = torch.cat([col, row], dim=0)
                asym_messages = torch.cat([edge_asym, -edge_asym], dim=0)
                asym_agg = scatter_add(asym_messages, undirected_edge_index,
                                       dim=0, dim_size=c_graph.num_nodes)

                # Combine all features
                collected_features = torch.cat([c_graph.x, cell_agg, asym_agg], dim=-1)
                cell_attr = self.cell_mlp(collected_features)

                return Data(x=cell_attr, edge_attr=c_graph.edge_attr, edge_index=c_graph.edge_index)

        class Face_Block(torch.nn.Module):
            def __init__(self, config, hidden_size):
                super().__init__()
                input_size = hidden_size * 2  # edge + two cells
                self.face_mlp = build_mlp(config, input_size, hidden_size, out_size=hidden_size)

            def forward(self, c_graph):
                row, col = c_graph.edge_index
                aggr_features = torch.cat([c_graph.edge_attr, c_graph.x[row] + c_graph.x[col]], dim=1)        # (E, F_x)
                edge_attr = self.face_mlp(aggr_features)
                return Data(x=c_graph.x, edge_attr=edge_attr, edge_index=c_graph.edge_index)


class ConservativeH(FvgnA):
    """
    Conservative message passing for Fvgn-like model with enforced
    symmetric/antisymmetric separation throughout the network.
    """
    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)

        # Encoder-Processor-Decoder
        self.encoder = self.Encoder(config, self.input_sizes, self.hidden_size)
        processer_list = []
        for _ in range(config.model.mp_num):
            processer_list.append(self.GN_Block(config, self.hidden_size))
        self.processer_list = torch.nn.ModuleList(processer_list)
        self.decoder = self.Decoder(config, self.hidden_size, self.output_sizes)

    @classmethod
    def transform_features(cls, dataset, graphs, mesh_id=None):
        cell_graph, face_graph, vertex_graph = graphs
        cell_velocity = cell_graph.velocity[:, 0]
        if dataset.noise:
            cell_velocity = transforms.add_noise(cell_velocity, std=dataset.config.training.noise_std)
        cell_graph.x = cell_velocity
        cell_graph.y = cell_graph.velocity[:, -1] - cell_velocity

        # Flip edges
        if dataset.mode == 'train':
            cell_graph.edge_index, flip_mask = transforms.random_edge_flip(cell_graph.edge_index)
            safe_flip = flip_mask & (cell_graph.edge_index[0] != cell_graph.edge_index[1]) # don't flip normal on boundaries
            face_graph.normal[safe_flip] *= -1

        # Use FVGN-like boundaries
        face_type = face_graph.type
        f_interior_mask = (face_type == dataset.class_types.NORMAL) | (face_type == dataset.class_types.OUTFLOW) | (face_type == dataset.class_types.SLIP) | (face_type == dataset.class_types.WALL_BOUNDARY)
        face_graph.boundary_mask = ~f_interior_mask.squeeze()

        face_velocity_change = transforms.calc_face_velocity_change(cell_velocity, cell_graph.edge_index)
        face_velocity_change[face_graph.boundary_mask] = face_graph.velocity[:, 0][face_graph.boundary_mask] # apply boundary conditions manually
        face_edge_vector  = transforms.calc_cell_edge_vector(cell_graph.pos, cell_graph.edge_index)
        face_type_one_hot = transforms.calc_face_type_one_hot(face_graph.type, len(dataset.class_types))
        face_graph.x_symm = torch.cat([face_graph.area, face_type_one_hot], dim=1)
        face_graph.x_asym = torch.cat([face_velocity_change, face_edge_vector], dim=1)
        face_graph.y = torch.cat([face_graph.velocity[:, -1], face_graph.pressure[:, -1]], dim=1)

        graphs = transforms.clean_graphs([cell_graph, face_graph, vertex_graph])

        return graphs

    @classmethod
    def get_normalisation_map(cls):
        registry = {
            # symm
            "cell_velocity_x": (lambda graphs: graphs[0].x[:, 0:1], 'z_score'),
            "cell_velocity_y": (lambda graphs: graphs[0].x[:, 1:2], 'z_score'),
            "cell_velocity_change_x": (lambda graphs: graphs[0].y[:, 0:1], 'z_score'),
            "cell_velocity_change_y": (lambda graphs: graphs[0].y[:, 1:2], 'z_score'),
            "face_area": (lambda graphs: graphs[1].x_symm[:, 0:1], 'z_score'),
            "face_velocity_x": (lambda graphs: graphs[1].y[:, 0:1], 'z_score'),
            "face_velocity_y": (lambda graphs: graphs[1].y[:, 1:2], 'z_score'),
            "face_pressure": (lambda graphs: graphs[1].y[:, 2:3], 'z_score'),
            # antisym
            "face_velocity_diff_x": (lambda graphs: graphs[1].x_asym[:, 0:1], 'std_scale'),
            "face_velocity_diff_y": (lambda graphs: graphs[1].x_asym[:, 1:2], 'std_scale'),
            "face_edge_vector_x": (lambda graphs: graphs[1].x_asym[:, 2:3], 'std_scale'),
            "face_edge_vector_y": (lambda graphs: graphs[1].x_asym[:, 3:4], 'std_scale'),
        }


        inputs = {
            # inputs
            "cell_velocity_x": (lambda graphs: graphs[0].x[:, 0:1], 'cell_velocity_x'),
            "cell_velocity_y": (lambda graphs: graphs[0].x[:, 1:2], 'cell_velocity_y'),
            "face_velocity_diff_x": (lambda graphs: graphs[1].x_asym[:, 0:1], 'face_velocity_diff_x'),
            "face_velocity_diff_y": (lambda graphs: graphs[1].x_asym[:, 1:2], 'face_velocity_diff_y'),
            "face_area": (lambda graphs: graphs[1].x_symm[:, 0:1], 'face_area'),
            "face_edge_vector_x": (lambda graphs: graphs[1].x_asym[:, 2:3], 'face_edge_vector_x'),
            "face_edge_vector_y": (lambda graphs: graphs[1].x_asym[:, 3:4], 'face_edge_vector_y'),
            # targets
            "cell_velocity_change_x": (lambda graphs: graphs[0].y[:, 0:1], 'cell_velocity_change_x'),
            "cell_velocity_change_y": (lambda graphs: graphs[0].y[:, 1:2], 'cell_velocity_change_y'),
            "face_velocity_x": (lambda graphs: graphs[1].y[:, 0:1], 'face_velocity_x'),
            "face_velocity_y": (lambda graphs: graphs[1].y[:, 1:2], 'face_velocity_y'),
            "face_pressure": (lambda graphs: graphs[1].y[:, 2:3], 'face_pressure')
        }

        outputs = {
            "cell_velocity_change_x": (lambda outputs: outputs[0][:, 0:1], 'cell_velocity_change_x'),
            "cell_velocity_change_y": (lambda outputs: outputs[0][:, 1:2], 'cell_velocity_change_y'),
            "face_velocity_x": (lambda outputs: outputs[1][:, 0:1], 'face_velocity_x'),
            "face_velocity_y": (lambda outputs: outputs[1][:, 1:2], 'face_velocity_y'),
            "face_pressure": (lambda outputs: outputs[1][:, 2:3], 'face_pressure')
        }

        return registry, inputs, outputs

    def update_features(self, output, input_graphs):
        """Update input graphs with predicted outputs for autoregressive rollout"""
        c_graph, f_graph, v_graph = input_graphs

        # Update cell velocity with predicted velocity
        c_graph.x = output["cell_velocity"].detach()

        face_velocity_change = transforms.calc_face_velocity_change(c_graph.x[:, :2], c_graph.edge_index)
        # mask = f_graph.boundary_mask
        face_type = f_graph.type
        mask = (face_type == NodeType.INFLOW) | (face_type == NodeType.WALL_BOUNDARY)
        face_velocity_change[mask] = f_graph.y[:, 0:2][mask] # apply boundary conditions manually

        f_graph.x_asym[:, 0:2] = face_velocity_change

        return [c_graph, f_graph, v_graph]

    @classmethod
    def get_feature_sizes(cls, dataset):
        return ([2, 1 + len(dataset.class_types), 0], [0, 5, 0])

    def forward(self, graphs, mode='rollout'):
        graphs = self.normalizer.input(graphs)
        c_graph, f_graph, v_graph = graphs
        c_graph.edge_attr = f_graph.x_symm
        c_graph.edge_attr_asym = f_graph.x_asym
        c_graph_geom = c_graph.clone()

        c_graph = self.encoder(c_graph)
        for gnblock in self.processer_list:
            c_graph = gnblock(c_graph, v_graph)
        edge_attr_out = self.decoder(c_graph)

        self.dt = c_graph_geom.dt
        acc_pred = self.integrator(edge_attr_out, c_graph_geom, f_graph, self.dt)
        output = [acc_pred, edge_attr_out, None]

        if mode == 'rollout':
            output = self.normalizer.output(output, inverse=True)

        return {
            'cell_velocity_change': output[0][:, 0:2],
            'face_velocity': output[1][:, :2],
            'face_pressure': output[1][:, 2:3],
        }


    class Integrator(torch.nn.Module):
        def __init__(self, config, rho):
            super().__init__()
            self.rho = rho
            self.face_area_norm = torch.nn.BatchNorm1d(1)
            self.face_area = None

        def forward(self, edge_output, c_graph, f_graph, dt):
            unv = c_graph.normal
            rhs_coef = 1.0
            face_area = f_graph.area
            cell_face = f_graph.face

            face_area = normalize_face_area(face_area, c_graph.volume, c_graph.edge_index, dt, self.face_area_norm)
            self.face_area = face_area

            e0 = face_area[cell_face[0]]
            e1 = face_area[cell_face[1]]
            e2 = face_area[cell_face[2]]

            uv_face = edge_output[:, :2]
            p_face = edge_output[:, 2:3]
            q_face = edge_output[:, 3:]

            uu_vu = torch.cat([uv_face[:, 0:1]*uv_face, uv_face[:, 1:2]*uv_face], dim=-1)

            A0 = chain_flux_dot_product(uu_vu[cell_face[0]], unv[:, 0, :]) * e0
            A1 = chain_flux_dot_product(uu_vu[cell_face[1]], unv[:, 1, :]) * e1
            A2 = chain_flux_dot_product(uu_vu[cell_face[2]], unv[:, 2, :]) * e2
            Phi_A = A0 + A1 + A2

            D0 = (q_face[cell_face[0]] * unv[:, 0, :]) * e0   # (B,2)
            D1 = (q_face[cell_face[1]] * unv[:, 1, :]) * e1
            D2 = (q_face[cell_face[2]] * unv[:, 2, :]) * e2
            Phi_D = D0 + D1 + D2

            P0 = p_face[cell_face[0]] * unv[:, 0, :] * e0
            P1 = p_face[cell_face[1]] * unv[:, 1, :] * e1
            P2 = p_face[cell_face[2]]* unv[:, 2, :] * e2
            Phi_P = P0 + P1 + P2

            return rhs_coef * (-Phi_A - Phi_P / self.rho) + Phi_D

    # Encoder and GN_Block are correctly implemented and do not need changes.
    class Encoder(torch.nn.Module):
        def __init__(self, config, input_sizes, hidden_size):
            super().__init__()
            self.faceA_mlp = build_mlp_antisym(config, 4, hidden_size, out_size=hidden_size)
            self.faceS_mlp = build_mlp(config, input_sizes[1], hidden_size, out_size=hidden_size)
            self.cell_mlp = build_mlp(config, input_sizes[0], hidden_size, out_size=hidden_size)

        def forward(self, cell_graph):
            face_attr_symm = self.faceS_mlp(cell_graph.edge_attr)
            face_attr_asym = self.faceA_mlp(cell_graph.edge_attr_asym)
            cell_attr = self.cell_mlp(cell_graph.x)
            return Data(x=cell_attr, edge_attr=face_attr_symm, edge_attr_asym=face_attr_asym, edge_index=cell_graph.edge_index)

    class GN_Block(torch.nn.Module):
        def __init__(self, config, hidden_size):
            super().__init__()
            self.face_block_symm = self.Face_Block_Symm(config, hidden_size)
            self.face_block_asym = self.Face_Block_Asym(config, hidden_size)
            self.cell_block = self.Cell_Block(config, hidden_size)

        def forward(self, c_graph, v_graph):
            prev_edge_attr_symm = c_graph.edge_attr.clone()
            prev_edge_attr_asym = c_graph.edge_attr_asym.clone()
            prev_node_attr = c_graph.x.clone()


            c_graph = self.cell_block(c_graph, v_graph)
            edge_attr_symm_updated = self.face_block_symm(c_graph)
            edge_attr_asym_updated = self.face_block_asym(c_graph)

            c_graph_updated = Data(
                x=c_graph.x,
                edge_attr=edge_attr_symm_updated,
                edge_attr_asym=edge_attr_asym_updated,
                edge_index=c_graph.edge_index
            )

            node_attr = prev_node_attr + c_graph_updated.x
            edge_attr_symm = prev_edge_attr_symm + c_graph_updated.edge_attr
            edge_attr_asym = prev_edge_attr_asym + c_graph_updated.edge_attr_asym

            return Data(x=node_attr, edge_attr=edge_attr_symm,
                        edge_attr_asym=edge_attr_asym, edge_index=c_graph.edge_index)

        class Face_Block_Symm(torch.nn.Module):
            def __init__(self, config, hidden_size):
                super().__init__()
                input_size = hidden_size * 2
                self.face_mlp = build_mlp(config, input_size, hidden_size, out_size=hidden_size)

            def forward(self, c_graph):
                row, col = c_graph.edge_index
                aggr_features = torch.cat([c_graph.edge_attr, c_graph.x[row] + c_graph.x[col]], dim=1)
                return self.face_mlp(aggr_features)

        class Face_Block_Asym(torch.nn.Module):
            def __init__(self, config, hidden_size):
                super().__init__()
                input_size = hidden_size * 2
                self.face_mlp = build_mlp_antisym(config, input_size, hidden_size, out_size=hidden_size)

            def forward(self, c_graph):
                row, col = c_graph.edge_index
                aggr_features = torch.cat([c_graph.edge_attr_asym, c_graph.x[row] - c_graph.x[col]], dim=1)
                return self.face_mlp(aggr_features)

        class Cell_Block(torch.nn.Module):
            def __init__(self, config, hidden_size):
                super().__init__()
                input_size = hidden_size * 3
                self.cell_mlp = build_mlp(config, input_size, hidden_size, out_size=hidden_size)

            def forward(self, c_graph, v_graph):
                edge_sym = c_graph.edge_attr
                edge_asym = c_graph.edge_attr_asym

                # Symmetric part aggregate via vertices
                senders_node_idx, receivers_node_idx = v_graph.edge_index
                twoway_node_connections = torch.cat([senders_node_idx, receivers_node_idx], dim=0)
                twoway_edge_attr = torch.cat([edge_sym, edge_sym], dim=0)
                node_agg_received_edges = scatter_add(twoway_edge_attr, twoway_node_connections,
                                                      dim=0, dim_size=v_graph.num_nodes)
                cell_agg = (
                    torch.index_select(node_agg_received_edges, 0, v_graph.face[0]) +
                    torch.index_select(node_agg_received_edges, 0, v_graph.face[1]) +
                    torch.index_select(node_agg_received_edges, 0, v_graph.face[2])
                ) / 3.0

                # Asymmetric part aggregate via edges
                row, col = c_graph.edge_index
                undirected_edge_index = torch.cat([col, row], dim=0)
                asym_messages = torch.cat([edge_asym, -edge_asym], dim=0)
                asym_agg = scatter_add(asym_messages, undirected_edge_index,
                                       dim=0, dim_size=c_graph.num_nodes)

                # Combine all features
                collected_features = torch.cat([c_graph.x, cell_agg, asym_agg], dim=-1)
                cell_attr = self.cell_mlp(collected_features)

                return Data(x=cell_attr, edge_attr=c_graph.edge_attr, edge_attr_asym=c_graph.edge_attr_asym, edge_index=c_graph.edge_index)

    class Decoder(nn.Module):
        def __init__(self, config, H, output_sizes):
            super().__init__()
            # Even head: (u_f, v_f, p_f, q_mag_u, q_mag_v) — all even
            self.even_mlp = build_mlp(config, in_size=2*H, hidden_size=H, out_size=5, norm_layer=False)

            # Odd head: s_odd_u, s_odd_v — both odd in h_minus
            self.odd_mlp  = build_mlp_antisym(config, in_size=H+H, hidden_size=H, out_size=2, norm_layer=False)

        def forward(self, graph):
            h_plus, h_minus = graph.edge_attr, graph.edge_attr_asym  # (E,H) each
            even_feats = torch.cat([h_plus, h_minus**2], dim=-1)     # even
            odd_feats  = torch.cat([h_minus, h_plus], dim=-1)        # carries sign

            uvp_qmag = self.even_mlp(even_feats)                     # (E,5)
            uv_face  = uvp_qmag[:, 0:2]
            p_face   = uvp_qmag[:, 2:3]
            q_mag    = torch.nn.functional.softplus(uvp_qmag[:, 3:5])                  # ≥0, even

            s_odd    = torch.tanh(self.odd_mlp(odd_feats))           # ∈(-1,1), odd

            q_n      = q_mag * s_odd                                 # signed normal flux
            return torch.cat([uv_face, p_face, q_n], dim=-1)         # [u,v,p,qn_u,qn_v]


class ConservativeI(FvgnA):
    """Add b.c.s"""
    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)

        processer_list = []
        for _ in range(config.model.mp_num):
            processer_list.append(self.GN_Block(config, self.hidden_size))
        self.processer_list = torch.nn.ModuleList(processer_list)

    def forward(self, graphs, mode='rollout'):
        graphs = self.normalizer.input(graphs)
        c_graph, f_graph, v_graph = graphs
        c_graph.edge_attr = f_graph.x
        c_graph_geom = c_graph.clone()

        c_graph = self.encoder(c_graph)

        for gnblock in self.processer_list:
            c_graph = gnblock(c_graph, v_graph, f_graph)

        edge_attr_out = self.decoder(c_graph)

        self.dt = c_graph_geom.dt
        acc_pred = self.integrator(edge_attr_out, c_graph_geom, f_graph, self.dt)
        output = [acc_pred, edge_attr_out, None]

        if mode == 'rollout':
            output = self.normalizer.output(output, inverse=True)

        return {
            'cell_velocity_change': output[0][:, 0:2],
            'face_velocity': output[1][:, :2],
            'face_pressure': output[1][:, 2:3],
        }

    class GN_Block(torch.nn.Module):
        def __init__(self, config, hidden_size):
            super().__init__()
            self.cell_block = self.Cell_Block(config, hidden_size)
            self.face_block = self.Face_Block(config, hidden_size)

        def forward(self, c_graph, v_graph, f_graph):
            prev_edge_attr = c_graph.edge_attr.clone()
            prev_node_attr = c_graph.x.clone()

            c_graph = self.cell_block(c_graph, v_graph)
            c_graph = self.face_block(c_graph)

            edge_attr = prev_edge_attr + c_graph.edge_attr
            node_attr = prev_node_attr + c_graph.x

            # Fix B.C.s
            face_type = f_graph.type
            mask = (face_type == NodeType.INFLOW) | (face_type == NodeType.WALL_BOUNDARY)
            edge_attr = edge_attr.clone()
            edge_attr[mask] = prev_edge_attr[mask]

            return Data(x=node_attr, edge_attr=edge_attr, edge_index=c_graph.edge_index)

        class Cell_Block(torch.nn.Module):
            def __init__(self, config, hidden_size):
                super().__init__()
                input_size = hidden_size * 2
                self.cell_mlp = build_mlp(config, input_size, hidden_size, out_size=hidden_size)
                # self.cell_mlp_asym = build_mlp_antisym(config, input_size, hidden_size, out_size=hidden_size//2)

            def forward(self, c_graph, v_graph):
                # edge_sym = c_graph.edge_attr_symm
                # edge_asym = c_graph.edge_attr_asym
                edge_sym, edge_asym = torch.chunk(c_graph.edge_attr, 2, dim=-1)

                # Symmetric part aggregate via vertices
                senders_node_idx, receivers_node_idx = v_graph.edge_index
                twoway_node_connections = torch.cat([senders_node_idx, receivers_node_idx], dim=0)
                twoway_edge_attr = torch.cat([edge_sym, edge_sym], dim=0)
                node_agg_received_edges = scatter_add(twoway_edge_attr, twoway_node_connections,
                                                      dim=0, dim_size=v_graph.num_nodes)
                cell_agg = (
                    torch.index_select(node_agg_received_edges, 0, v_graph.face[0]) +
                    torch.index_select(node_agg_received_edges, 0, v_graph.face[1]) +
                    torch.index_select(node_agg_received_edges, 0, v_graph.face[2])
                ) / 3.0

                row, col = c_graph.edge_index
                undirected_edge_index = torch.cat([col, row], dim=0)
                asym_messages = torch.cat([edge_asym, -edge_asym], dim=0)
                asym_agg = scatter_add(asym_messages, undirected_edge_index,
                                       dim=0, dim_size=c_graph.num_nodes)

                # Combine all features
                collected_features = torch.cat([c_graph.x, cell_agg, asym_agg], dim=-1)
                cell_attr = self.cell_mlp(collected_features)

                return Data(x=cell_attr, edge_attr=c_graph.edge_attr, edge_index=c_graph.edge_index)

        class Face_Block(torch.nn.Module):
            def __init__(self, config, hidden_size):
                super().__init__()
                input_size = hidden_size * 2  # edge + two cells
                self.face_mlp = build_mlp(config, input_size, hidden_size, out_size=hidden_size)

            def forward(self, c_graph):
                row, col = c_graph.edge_index
                aggr_features = torch.cat([c_graph.edge_attr, c_graph.x[row] + c_graph.x[col]], dim=1)        # (E, F_x)
                edge_attr = self.face_mlp(aggr_features)
                return Data(x=c_graph.x, edge_attr=edge_attr, edge_index=c_graph.edge_index)


class ConservativeJ(FvgnA):
    """
    Conservative message passing for Fvgn-like model with enforced
    symmetric/antisymmetric separation throughout the network.
    """
    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)

        # Encoder-Processor-Decoder
        self.encoder = self.Encoder(config, self.input_sizes, self.hidden_size)
        processer_list = []
        for _ in range(config.model.mp_num):
            processer_list.append(self.GN_Block(config, self.hidden_size))
        self.processer_list = torch.nn.ModuleList(processer_list)
        self.decoder = self.Decoder(config, self.hidden_size, self.output_sizes)

        self.velocity_scale_x = torch.nn.Parameter(torch.tensor(1.0))
        self.velocity_scale_y = torch.nn.Parameter(torch.tensor(0.01))
        self.pressure_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.diffusion_scale = torch.nn.Parameter(torch.tensor(1.0))

        self.velocity_bias_x = torch.nn.Parameter(torch.tensor(0.0))
        self.velocity_bias_y = torch.nn.Parameter(torch.tensor(0.0))
        self.pressure_bias = torch.nn.Parameter(torch.tensor(0.0))

    @classmethod
    def transform_features(cls, dataset, graphs, mesh_id=None):
        cell_graph, face_graph, vertex_graph = graphs
        cell_velocity = cell_graph.velocity[:, 0]
        if dataset.noise:
            cell_velocity = transforms.add_noise(cell_velocity, std=dataset.config.training.noise_std)
        cell_graph.x = cell_velocity
        cell_graph.y = cell_graph.velocity[:, -1] - cell_velocity

        # Flip edges
        if dataset.mode == 'train':
            cell_graph.edge_index, flip_mask = transforms.random_edge_flip(cell_graph.edge_index)
            safe_flip = flip_mask & (cell_graph.edge_index[0] != cell_graph.edge_index[1]) # don't flip normal on boundaries
            face_graph.normal[safe_flip] *= -1

        # Use FVGN-like boundaries
        face_type = face_graph.type
        f_interior_mask = (face_type == dataset.class_types.NORMAL) | (face_type == dataset.class_types.OUTFLOW) | (face_type == dataset.class_types.SLIP) | (face_type == dataset.class_types.WALL_BOUNDARY)
        face_graph.boundary_mask = ~f_interior_mask.squeeze()

        face_velocity_change = transforms.calc_face_velocity_change(cell_velocity, cell_graph.edge_index)
        face_velocity_change[face_graph.boundary_mask] = face_graph.velocity[:, 0][face_graph.boundary_mask] # apply boundary conditions manually
        face_edge_vector  = transforms.calc_cell_edge_vector(cell_graph.pos, cell_graph.edge_index)
        face_type_one_hot = transforms.calc_face_type_one_hot(face_graph.type, len(dataset.class_types))
        face_graph.x_symm = torch.cat([face_graph.area, face_type_one_hot], dim=1)
        face_graph.x_asym = torch.cat([face_velocity_change, face_edge_vector], dim=1)
        face_graph.y = torch.cat([face_graph.velocity[:, -1], face_graph.pressure[:, -1]], dim=1)

        graphs = transforms.clean_graphs([cell_graph, face_graph, vertex_graph])

        return graphs

    @classmethod
    def get_normalisation_map(cls):
        registry = {
            # symm
            "cell_velocity_x": (lambda graphs: graphs[0].x[:, 0:1], 'z_score'),
            "cell_velocity_y": (lambda graphs: graphs[0].x[:, 1:2], 'z_score'),
            "cell_velocity_change_x": (lambda graphs: graphs[0].y[:, 0:1], 'z_score'),
            "cell_velocity_change_y": (lambda graphs: graphs[0].y[:, 1:2], 'z_score'),
            "face_area": (lambda graphs: graphs[1].x_symm[:, 0:1], 'z_score'),
            "face_velocity_x": (lambda graphs: graphs[1].y[:, 0:1], 'z_score'),
            "face_velocity_y": (lambda graphs: graphs[1].y[:, 1:2], 'z_score'),
            "face_pressure": (lambda graphs: graphs[1].y[:, 2:3], 'z_score'),
            # antisym
            "face_velocity_diff_x": (lambda graphs: graphs[1].x_asym[:, 0:1], 'std_scale'),
            "face_velocity_diff_y": (lambda graphs: graphs[1].x_asym[:, 1:2], 'std_scale'),
            "face_edge_vector_x": (lambda graphs: graphs[1].x_asym[:, 2:3], 'std_scale'),
            "face_edge_vector_y": (lambda graphs: graphs[1].x_asym[:, 3:4], 'std_scale'),
        }


        inputs = {
            # inputs
            "cell_velocity_x": (lambda graphs: graphs[0].x[:, 0:1], 'cell_velocity_x'),
            "cell_velocity_y": (lambda graphs: graphs[0].x[:, 1:2], 'cell_velocity_y'),
            "face_velocity_diff_x": (lambda graphs: graphs[1].x_asym[:, 0:1], 'face_velocity_diff_x'),
            "face_velocity_diff_y": (lambda graphs: graphs[1].x_asym[:, 1:2], 'face_velocity_diff_y'),
            "face_area": (lambda graphs: graphs[1].x_symm[:, 0:1], 'face_area'),
            "face_edge_vector_x": (lambda graphs: graphs[1].x_asym[:, 2:3], 'face_edge_vector_x'),
            "face_edge_vector_y": (lambda graphs: graphs[1].x_asym[:, 3:4], 'face_edge_vector_y'),
            # targets
            "cell_velocity_change_x": (lambda graphs: graphs[0].y[:, 0:1], 'cell_velocity_change_x'),
            "cell_velocity_change_y": (lambda graphs: graphs[0].y[:, 1:2], 'cell_velocity_change_y'),
            "face_velocity_x": (lambda graphs: graphs[1].y[:, 0:1], 'face_velocity_x'),
            "face_velocity_y": (lambda graphs: graphs[1].y[:, 1:2], 'face_velocity_y'),
            "face_pressure": (lambda graphs: graphs[1].y[:, 2:3], 'face_pressure')
        }

        outputs = {
            "cell_velocity_change_x": (lambda outputs: outputs[0][:, 0:1], 'cell_velocity_change_x'),
            "cell_velocity_change_y": (lambda outputs: outputs[0][:, 1:2], 'cell_velocity_change_y'),
            "face_velocity_x": (lambda outputs: outputs[1][:, 0:1], 'face_velocity_x'),
            "face_velocity_y": (lambda outputs: outputs[1][:, 1:2], 'face_velocity_y'),
            "face_pressure": (lambda outputs: outputs[1][:, 2:3], 'face_pressure')
        }

        return registry, inputs, outputs

    def update_features(self, output, input_graphs):
        """Update input graphs with predicted outputs for autoregressive rollout"""
        c_graph, f_graph, v_graph = input_graphs

        # Update cell velocity with predicted velocity
        c_graph.x = output["cell_velocity"].detach()

        face_velocity_change = transforms.calc_face_velocity_change(c_graph.x[:, :2], c_graph.edge_index)
        # mask = f_graph.boundary_mask
        face_type = f_graph.type
        mask = (face_type == NodeType.INFLOW) | (face_type == NodeType.WALL_BOUNDARY)
        face_velocity_change[mask] = f_graph.y[:, 0:2][mask] # apply boundary conditions manually

        f_graph.x_asym[:, 0:2] = face_velocity_change

        return [c_graph, f_graph, v_graph]

    def loss(self, output, graphs):
        c_graph, f_graph, v_graph = graphs
        loss_func = self.loss_func

        face_area = f_graph.x_symm[:, 0:1] # normalised face_area
        cell_divergence = fvm.divergence_from_uf(output['face_velocity'], c_graph.normal, face_area, f_graph.face)
        continuity   = loss_func(cell_divergence,
                                        torch.zeros_like(cell_divergence),
                                        None,
                                        c_graph.batch)

        cell_velocity_change     = loss_func(output['cell_velocity_change'],
                                        c_graph.y,
                                        None,
                                        c_graph.batch)

        face_velocity = loss_func(output['face_velocity'],
                                        f_graph.y[:, :2],
                                        ~f_graph.boundary_mask, # only interior
                                        f_graph.batch)

        face_pressure = loss_func(output['face_pressure'],
                                        f_graph.y[:, 2:3],
                                        None,
                                        f_graph.batch)

        w = self.config.training.loss_weights
        total = w['continuity'] * continuity + w['cell_velocity_change'] * cell_velocity_change + w['face_velocity'] * face_velocity + w['face_pressure'] * face_pressure
        loss = torch.mean(torch.log(total))

        return {
            "total_log_loss": loss,
            "continuity_loss": continuity,
            "cell_velocity_change_loss": cell_velocity_change,
            "face_velocity_loss": face_velocity,
            "face_pressure_loss": face_pressure
        }

    @classmethod
    def get_feature_sizes(cls, dataset):
        return ([2, 1 + len(dataset.class_types), 0], [0, 5, 0])

    def forward(self, graphs, mode='rollout'):
        graphs = self.normalizer.input(graphs)
        c_graph, f_graph, v_graph = graphs
        c_graph.edge_attr = f_graph.x_symm
        c_graph.edge_attr_asym = f_graph.x_asym
        c_graph_geom = c_graph.clone()

        c_graph = self.encoder(c_graph)
        for gnblock in self.processer_list:
            c_graph = gnblock(c_graph, v_graph)

        edge_attr_out_raw = self.decoder(c_graph) # ouputs normalised

        u_face = edge_attr_out_raw[:, 0:1] * self.velocity_scale_x + self.velocity_bias_x
        v_face = edge_attr_out_raw[:, 1:2] * self.velocity_scale_y + self.velocity_bias_y
        uv_face = torch.cat([u_face, v_face], dim=-1)
        p_face = edge_attr_out_raw[:, 2:3] * self.pressure_scale + self.pressure_bias
        d_flux = edge_attr_out_raw[:, 3:5] * self.diffusion_scale

        edge_attr_out = torch.cat([uv_face, p_face, d_flux], dim=-1)
        output = [None, edge_attr_out, None]

        # output = self.normalizer.output([None, edge_attr_out.clone(), None], inverse=True) # denormalise for integrator
        self.dt = c_graph_geom.dt
        acc_pred = self.integrator(output[1], c_graph_geom, f_graph, self.dt) # outputs denormalised

        output = [acc_pred, edge_attr_out, None]
        if mode != 'rollout': # normalised for training
            output = self.normalizer.output(output) # normalise for loss

        return {
            'cell_velocity_change': output[0][:, 0:2],
            'face_velocity': output[1][:, :2],
            'face_pressure': output[1][:, 2:3],
        }


    class Integrator(torch.nn.Module):
        def __init__(self, config, rho):
            super().__init__()
            self.rho = rho
            self.nu = 0.001

        def forward(self, edge_output, c_graph, f_graph, dt):
            unv = c_graph.normal
            face_area = f_graph.area
            cell_face = f_graph.face

            e0 = face_area[cell_face[0]]
            e1 = face_area[cell_face[1]]
            e2 = face_area[cell_face[2]]

            uv_face = edge_output[:, 0:2].clone()
            p_face = edge_output[:, 2:3].clone()
            q_face = edge_output[:, 3:5].clone()

            uu_vu = torch.cat([uv_face[:, 0:1]*uv_face, uv_face[:, 1:2]*uv_face], dim=-1)

            A0 = chain_flux_dot_product(uu_vu[cell_face[0]], unv[:, 0, :]) * e0
            A1 = chain_flux_dot_product(uu_vu[cell_face[1]], unv[:, 1, :]) * e1
            A2 = chain_flux_dot_product(uu_vu[cell_face[2]], unv[:, 2, :]) * e2
            Phi_A = A0 + A1 + A2

            D0 = (q_face[cell_face[0]]* unv[:, 0, :]) * e0   # (B,2)
            D1 = (q_face[cell_face[1]]* unv[:, 1, :]) * e1
            D2 = (q_face[cell_face[2]]* unv[:, 2, :]) * e2
            Phi_D = D0 + D1 + D2

            P0 = p_face[cell_face[0]] * unv[:, 0, :] * e0
            P1 = p_face[cell_face[1]] * unv[:, 1, :] * e1
            P2 = p_face[cell_face[2]]* unv[:, 2, :] * e2
            Phi_P = P0 + P1 + P2

            return torch.mean(dt) / c_graph.volume * (-Phi_A - Phi_P / self.rho + self.nu * Phi_D )

    # Encoder and GN_Block are correctly implemented and do not need changes.
    class Encoder(torch.nn.Module):
        def __init__(self, config, input_sizes, hidden_size):
            super().__init__()
            self.faceA_mlp = build_mlp_antisym(config, 4, hidden_size, out_size=hidden_size)
            self.faceS_mlp = build_mlp(config, input_sizes[1], hidden_size, out_size=hidden_size)
            self.cell_mlp = build_mlp(config, input_sizes[0], hidden_size, out_size=hidden_size)

        def forward(self, cell_graph):
            face_attr_symm = self.faceS_mlp(cell_graph.edge_attr)
            face_attr_asym = self.faceA_mlp(cell_graph.edge_attr_asym)
            cell_attr = self.cell_mlp(cell_graph.x)
            return Data(x=cell_attr, edge_attr=face_attr_symm, edge_attr_asym=face_attr_asym, edge_index=cell_graph.edge_index)

    class GN_Block(torch.nn.Module):
        def __init__(self, config, hidden_size):
            super().__init__()
            self.face_block_symm = self.Face_Block_Symm(config, hidden_size)
            self.face_block_asym = self.Face_Block_Asym(config, hidden_size)
            self.cell_block = self.Cell_Block(config, hidden_size)

        def forward(self, c_graph, v_graph):
            prev_edge_attr_symm = c_graph.edge_attr.clone()
            prev_edge_attr_asym = c_graph.edge_attr_asym.clone()
            prev_node_attr = c_graph.x.clone()


            c_graph = self.cell_block(c_graph, v_graph)
            edge_attr_symm_updated = self.face_block_symm(c_graph)
            edge_attr_asym_updated = self.face_block_asym(c_graph)

            c_graph_updated = Data(
                x=c_graph.x,
                edge_attr=edge_attr_symm_updated,
                edge_attr_asym=edge_attr_asym_updated,
                edge_index=c_graph.edge_index
            )

            node_attr = prev_node_attr + c_graph_updated.x
            edge_attr_symm = prev_edge_attr_symm + c_graph_updated.edge_attr
            edge_attr_asym = prev_edge_attr_asym + c_graph_updated.edge_attr_asym

            return Data(x=node_attr, edge_attr=edge_attr_symm,
                        edge_attr_asym=edge_attr_asym, edge_index=c_graph.edge_index)

        class Face_Block_Symm(torch.nn.Module):
            def __init__(self, config, hidden_size):
                super().__init__()
                input_size = hidden_size * 2
                self.face_mlp = build_mlp(config, input_size, hidden_size, out_size=hidden_size)

            def forward(self, c_graph):
                row, col = c_graph.edge_index
                aggr_features = torch.cat([c_graph.edge_attr, c_graph.x[row] + c_graph.x[col]], dim=1)
                return self.face_mlp(aggr_features)

        class Face_Block_Asym(torch.nn.Module):
            def __init__(self, config, hidden_size):
                super().__init__()
                input_size = hidden_size * 2
                self.face_mlp = build_mlp_antisym(config, input_size, hidden_size, out_size=hidden_size)

            def forward(self, c_graph):
                row, col = c_graph.edge_index
                aggr_features = torch.cat([c_graph.edge_attr_asym, c_graph.x[row] - c_graph.x[col]], dim=1)
                return self.face_mlp(aggr_features)

        class Cell_Block(torch.nn.Module):
            def __init__(self, config, hidden_size):
                super().__init__()
                input_size = hidden_size * 3
                self.cell_mlp = build_mlp(config, input_size, hidden_size, out_size=hidden_size)

            def forward(self, c_graph, v_graph):
                edge_sym = c_graph.edge_attr
                edge_asym = c_graph.edge_attr_asym

                # Symmetric part aggregate via vertices
                senders_node_idx, receivers_node_idx = v_graph.edge_index
                twoway_node_connections = torch.cat([senders_node_idx, receivers_node_idx], dim=0)
                twoway_edge_attr = torch.cat([edge_sym, edge_sym], dim=0)
                node_agg_received_edges = scatter_add(twoway_edge_attr, twoway_node_connections,
                                                      dim=0, dim_size=v_graph.num_nodes)
                cell_agg = (
                    torch.index_select(node_agg_received_edges, 0, v_graph.face[0]) +
                    torch.index_select(node_agg_received_edges, 0, v_graph.face[1]) +
                    torch.index_select(node_agg_received_edges, 0, v_graph.face[2])
                ) / 3.0

                # Asymmetric part aggregate via edges
                row, col = c_graph.edge_index
                undirected_edge_index = torch.cat([col, row], dim=0)
                asym_messages = torch.cat([edge_asym, -edge_asym], dim=0)
                asym_agg = scatter_add(asym_messages, undirected_edge_index,
                                       dim=0, dim_size=c_graph.num_nodes)

                # Combine all features
                collected_features = torch.cat([c_graph.x, cell_agg, asym_agg], dim=-1)
                cell_attr = self.cell_mlp(collected_features)

                return Data(x=cell_attr, edge_attr=c_graph.edge_attr, edge_attr_asym=c_graph.edge_attr_asym, edge_index=c_graph.edge_index)

    class Decoder(nn.Module):
        def __init__(self, config, H, output_sizes):
            super().__init__()
            # Even head: (u_f, v_f, p_f, q_mag_u, q_mag_v) — all even
            self.even_mlp = build_mlp(config, in_size=2*H, hidden_size=H, out_size=5, norm_layer=False)

            # Odd head: s_odd_u, s_odd_v — both odd in h_minus
            self.odd_mlp  = build_mlp_antisym(config, in_size=H+H, hidden_size=H, out_size=2, norm_layer=False)

        def forward(self, graph):
            h_plus, h_minus = graph.edge_attr, graph.edge_attr_asym  # (E,H) each
            even_feats = torch.cat([h_plus, h_minus**2], dim=-1)     # even
            odd_feats  = torch.cat([h_minus, h_plus], dim=-1)        # carries sign

            uvp_qmag = self.even_mlp(even_feats)                     # (E,5)
            uv_face  = uvp_qmag[:, 0:2]
            p_face   = uvp_qmag[:, 2:3]
            q_mag    = torch.nn.functional.softplus(uvp_qmag[:, 3:5])                  # ≥0, even

            s_odd    = torch.tanh(self.odd_mlp(odd_feats))           # ∈(-1,1), odd

            q_n      = q_mag * s_odd                                 # signed normal flux
            return torch.cat([uv_face, p_face, q_n], dim=-1)         # [u,v,p,qn_u,qn_v]


class ConservativeK(FvgnA):
    """
    Conservative message passing with antisymmetric branch at half hidden width.
    Symmetric edge channels: H
    Antisymmetric edge channels: H//2
    """
    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)
        assert self.hidden_size % 2 == 0, "hidden_size must be even to halve antisymmetric width"
        self.h_half = self.hidden_size // 2

        # Encoder-Processor-Decoder
        self.encoder = self.Encoder(config, self.input_sizes, self.hidden_size)
        processer_list = []
        for _ in range(config.model.mp_num):
            processer_list.append(self.GN_Block(config, self.hidden_size))
        self.processer_list = torch.nn.ModuleList(processer_list)
        self.decoder = self.Decoder(config, self.hidden_size, self.output_sizes)

    @classmethod
    def transform_features(cls, dataset, graphs, mesh_id=None):
        cell_graph, face_graph, vertex_graph = graphs
        cell_velocity = cell_graph.velocity[:, 0]
        if dataset.noise:
            cell_velocity = transforms.add_noise(cell_velocity, std=dataset.config.training.noise_std)
        cell_graph.x = cell_velocity
        cell_graph.y = cell_graph.velocity[:, -1] - cell_velocity

        if dataset.mode == 'train':
            cell_graph.edge_index, flip_mask = transforms.random_edge_flip(cell_graph.edge_index)
            safe_flip = flip_mask & (cell_graph.edge_index[0] != cell_graph.edge_index[1])
            face_graph.normal[safe_flip] *= -1

        face_type = face_graph.type
        f_interior_mask = (
            (face_type == dataset.class_types.NORMAL) |
            (face_type == dataset.class_types.OUTFLOW) |
            (face_type == dataset.class_types.SLIP) |
            (face_type == dataset.class_types.WALL_BOUNDARY)
        )
        face_graph.boundary_mask = ~f_interior_mask.squeeze()

        face_velocity_change = transforms.calc_face_velocity_change(cell_velocity, cell_graph.edge_index)
        face_velocity_change[face_graph.boundary_mask] = face_graph.velocity[:, 0][face_graph.boundary_mask]
        face_edge_vector = transforms.calc_cell_edge_vector(cell_graph.pos, cell_graph.edge_index)
        face_type_one_hot = transforms.calc_face_type_one_hot(face_graph.type, len(dataset.class_types))
        face_graph.x_symm = torch.cat([face_graph.area, face_type_one_hot], dim=1)
        face_graph.x_asym = torch.cat([face_velocity_change, face_edge_vector], dim=1)
        face_graph.y = torch.cat([face_graph.velocity[:, -1], face_graph.pressure[:, -1]], dim=1)

        graphs = transforms.clean_graphs([cell_graph, face_graph, vertex_graph])
        return graphs

    @classmethod
    def get_normalisation_map(cls):
        registry = {
            "cell_velocity_x": (lambda g: g[0].x[:, 0:1], 'z_score'),
            "cell_velocity_y": (lambda g: g[0].x[:, 1:2], 'z_score'),
            "cell_velocity_change_x": (lambda g: g[0].y[:, 0:1], 'z_score'),
            "cell_velocity_change_y": (lambda g: g[0].y[:, 1:2], 'z_score'),
            "face_area": (lambda g: g[1].x_symm[:, 0:1], 'z_score'),
            "face_velocity_x": (lambda g: g[1].y[:, 0:1], 'z_score'),
            "face_velocity_y": (lambda g: g[1].y[:, 1:2], 'z_score'),
            "face_pressure": (lambda g: g[1].y[:, 2:3], 'z_score'),
            "face_velocity_diff_x": (lambda g: g[1].x_asym[:, 0:1], 'std_scale'),
            "face_velocity_diff_y": (lambda g: g[1].x_asym[:, 1:2], 'std_scale'),
            "face_edge_vector_x": (lambda g: g[1].x_asym[:, 2:3], 'std_scale'),
            "face_edge_vector_y": (lambda g: g[1].x_asym[:, 3:4], 'std_scale'),
        }
        inputs = {
            "cell_velocity_x": (lambda g: g[0].x[:, 0:1], 'cell_velocity_x'),
            "cell_velocity_y": (lambda g: g[0].x[:, 1:2], 'cell_velocity_y'),
            "face_velocity_diff_x": (lambda g: g[1].x_asym[:, 0:1], 'face_velocity_diff_x'),
            "face_velocity_diff_y": (lambda g: g[1].x_asym[:, 1:2], 'face_velocity_diff_y'),
            "face_area": (lambda g: g[1].x_symm[:, 0:1], 'face_area'),
            "face_edge_vector_x": (lambda g: g[1].x_asym[:, 2:3], 'face_edge_vector_x'),
            "face_edge_vector_y": (lambda g: g[1].x_asym[:, 3:4], 'face_edge_vector_y'),
            "cell_velocity_change_x": (lambda g: g[0].y[:, 0:1], 'cell_velocity_change_x'),
            "cell_velocity_change_y": (lambda g: g[0].y[:, 1:2], 'cell_velocity_change_y'),
            "face_velocity_x": (lambda g: g[1].y[:, 0:1], 'face_velocity_x'),
            "face_velocity_y": (lambda g: g[1].y[:, 1:2], 'face_velocity_y'),
            "face_pressure": (lambda g: g[1].y[:, 2:3], 'face_pressure')
        }
        outputs = {
            "cell_velocity_change_x": (lambda o: o[0][:, 0:1], 'cell_velocity_change_x'),
            "cell_velocity_change_y": (lambda o: o[0][:, 1:2], 'cell_velocity_change_y'),
            "face_velocity_x": (lambda o: o[1][:, 0:1], 'face_velocity_x'),
            "face_velocity_y": (lambda o: o[1][:, 1:2], 'face_velocity_y'),
            "face_pressure": (lambda o: o[1][:, 2:3], 'face_pressure')
        }
        return registry, inputs, outputs

    def update_features(self, output, input_graphs):
        c_graph, f_graph, v_graph = input_graphs
        c_graph.x = output["cell_velocity"].detach()
        face_velocity_change = transforms.calc_face_velocity_change(c_graph.x[:, :2], c_graph.edge_index)
        face_type = f_graph.type
        mask = (face_type == NodeType.INFLOW) | (face_type == NodeType.WALL_BOUNDARY)
        face_velocity_change[mask] = f_graph.y[:, 0:2][mask]
        f_graph.x_asym[:, 0:2] = face_velocity_change
        return [c_graph, f_graph, v_graph]

    @classmethod
    def get_feature_sizes(cls, dataset):
        return ([2, 1 + len(dataset.class_types), 0], [0, 5, 0])

    def forward(self, graphs, mode='rollout'):
        graphs = self.normalizer.input(graphs)
        c_graph, f_graph, v_graph = graphs
        c_graph.edge_attr = f_graph.x_symm
        c_graph.edge_attr_asym = f_graph.x_asym
        c_graph_geom = c_graph.clone()

        c_graph = self.encoder(c_graph)
        for gnblock in self.processer_list:
            c_graph = gnblock(c_graph, v_graph)
        edge_attr_out = self.decoder(c_graph)

        self.dt = c_graph_geom.dt
        acc_pred = self.integrator(edge_attr_out, c_graph_geom, f_graph, self.dt)
        output = [acc_pred, edge_attr_out, None]
        if mode == 'rollout':
            output = self.normalizer.output(output, inverse=True)
        return {
            'cell_velocity_change': output[0][:, 0:2],
            'face_velocity': output[1][:, :2],
            'face_pressure': output[1][:, 2:3],
        }

    class Integrator(torch.nn.Module):
        def __init__(self, config, rho):
            super().__init__()
            self.rho = rho
            self.face_area_norm = torch.nn.BatchNorm1d(1)
            self.face_area = None
        def forward(self, edge_output, c_graph, f_graph, dt):
            unv = c_graph.normal
            rhs_coef = 1.0
            face_area = f_graph.area
            cell_face = f_graph.face
            face_area = normalize_face_area(face_area, c_graph.volume, c_graph.edge_index, dt, self.face_area_norm)
            self.face_area = face_area
            e0 = face_area[cell_face[0]]
            e1 = face_area[cell_face[1]]
            e2 = face_area[cell_face[2]]
            uv_face = edge_output[:, :2]
            p_face = edge_output[:, 2:3]
            q_face = edge_output[:, 3:]
            uu_vu = torch.cat([uv_face[:, 0:1]*uv_face, uv_face[:, 1:2]*uv_face], dim=-1)
            A0 = chain_flux_dot_product(uu_vu[cell_face[0]], unv[:, 0, :]) * e0
            A1 = chain_flux_dot_product(uu_vu[cell_face[1]], unv[:, 1, :]) * e1
            A2 = chain_flux_dot_product(uu_vu[cell_face[2]], unv[:, 2, :]) * e2
            Phi_A = A0 + A1 + A2
            D0 = (q_face[cell_face[0]] * unv[:, 0, :]) * e0
            D1 = (q_face[cell_face[1]] * unv[:, 1, :]) * e1
            D2 = (q_face[cell_face[2]] * unv[:, 2, :]) * e2
            Phi_D = D0 + D1 + D2
            P0 = p_face[cell_face[0]] * unv[:, 0, :] * e0
            P1 = p_face[cell_face[1]] * unv[:, 1, :] * e1
            P2 = p_face[cell_face[2]] * unv[:, 2, :] * e2
            Phi_P = P0 + P1 + P2
            return rhs_coef * (-Phi_A - Phi_P / self.rho) + Phi_D

    class Encoder(torch.nn.Module):
        def __init__(self, config, input_sizes, hidden_size):
            super().__init__()
            h_half = hidden_size // 2
            self.faceA_mlp = build_mlp_antisym(config, 4, hidden_size, out_size=h_half)
            self.faceS_mlp = build_mlp(config, input_sizes[1], hidden_size, out_size=hidden_size)
            self.cell_mlp = build_mlp(config, input_sizes[0], hidden_size, out_size=hidden_size)
        def forward(self, cell_graph):
            face_attr_symm = self.faceS_mlp(cell_graph.edge_attr)        # (E,H)
            face_attr_asym = self.faceA_mlp(cell_graph.edge_attr_asym)   # (E,H//2)
            cell_attr = self.cell_mlp(cell_graph.x)                      # (N,H)
            return Data(x=cell_attr, edge_attr=face_attr_symm,
                        edge_attr_asym=face_attr_asym, edge_index=cell_graph.edge_index)

    class GN_Block(torch.nn.Module):
        def __init__(self, config, hidden_size):
            super().__init__()
            self.hidden_size = hidden_size
            self.h_half = hidden_size // 2
            self.face_block_symm = self.Face_Block_Symm(config, hidden_size)
            self.face_block_asym = self.Face_Block_Asym(config, hidden_size)
            self.cell_block = self.Cell_Block(config, hidden_size)
        def forward(self, c_graph, v_graph):
            prev_edge_attr_symm = c_graph.edge_attr.clone()      # (E,H)
            prev_edge_attr_asym = c_graph.edge_attr_asym.clone() # (E,H//2)
            prev_node_attr = c_graph.x.clone()
            c_graph = self.cell_block(c_graph, v_graph)
            edge_attr_symm_updated = self.face_block_symm(c_graph)    # (E,H)
            edge_attr_asym_updated = self.face_block_asym(c_graph)    # (E,H//2)
            node_attr = prev_node_attr + c_graph.x
            edge_attr_symm = prev_edge_attr_symm + edge_attr_symm_updated
            edge_attr_asym = prev_edge_attr_asym + edge_attr_asym_updated
            return Data(x=node_attr, edge_attr=edge_attr_symm,
                        edge_attr_asym=edge_attr_asym, edge_index=c_graph.edge_index)

        class Face_Block_Symm(torch.nn.Module):
            def __init__(self, config, hidden_size):
                super().__init__()
                input_size = hidden_size * 2
                self.face_mlp = build_mlp(config, input_size, hidden_size, out_size=hidden_size)
            def forward(self, c_graph):
                row, col = c_graph.edge_index
                aggr = torch.cat([c_graph.edge_attr, c_graph.x[row] + c_graph.x[col]], dim=1)
                return self.face_mlp(aggr)

        class Face_Block_Asym(torch.nn.Module):
            def __init__(self, config, hidden_size):
                super().__init__()
                h_half = hidden_size // 2
                input_size = h_half + hidden_size          # (H//2 + H)
                self.face_mlp = build_mlp_antisym(config, input_size, hidden_size, out_size=h_half)
            def forward(self, c_graph):
                row, col = c_graph.edge_index
                aggr = torch.cat([c_graph.edge_attr_asym, c_graph.x[row] - c_graph.x[col]], dim=1)
                return self.face_mlp(aggr)

        class Cell_Block(torch.nn.Module):
            def __init__(self, config, hidden_size):
                super().__init__()
                h_half = hidden_size // 2
                input_size = hidden_size + hidden_size + h_half  # x + symm + asym
                self.cell_mlp = build_mlp(config, input_size, hidden_size, out_size=hidden_size)
            def forward(self, c_graph, v_graph):
                edge_sym = c_graph.edge_attr          # (E,H)
                edge_asym = c_graph.edge_attr_asym    # (E,H//2)

                # Symmetric aggregation via vertices
                send_i, recv_i = v_graph.edge_index
                tw_nodes = torch.cat([send_i, recv_i], dim=0)
                tw_edges = torch.cat([edge_sym, edge_sym], dim=0)
                node_agg = scatter_add(tw_edges, tw_nodes, dim=0, dim_size=v_graph.num_nodes)
                cell_agg = (
                    torch.index_select(node_agg, 0, v_graph.face[0]) +
                    torch.index_select(node_agg, 0, v_graph.face[1]) +
                    torch.index_select(node_agg, 0, v_graph.face[2])
                ) / 3.0

                # Asymmetric aggregation via edges
                row, col = c_graph.edge_index
                undirected_idx = torch.cat([col, row], dim=0)
                asym_msgs = torch.cat([edge_asym, -edge_asym], dim=0)
                asym_agg = scatter_add(asym_msgs, undirected_idx, dim=0, dim_size=c_graph.num_nodes)

                collected = torch.cat([c_graph.x, cell_agg, asym_agg], dim=-1)
                cell_attr = self.cell_mlp(collected)
                return Data(x=cell_attr, edge_attr=edge_sym,
                            edge_attr_asym=edge_asym, edge_index=c_graph.edge_index)

    class Decoder(nn.Module):
        def __init__(self, config, H, output_sizes):
            super().__init__()
            h_half = H // 2
            # even features: h_plus (H) + (h_minus^2) (H/2) = 1.5H
            self.even_mlp = build_mlp(config, in_size=H + h_half, hidden_size=H, out_size=5, norm_layer=False)
            # odd features: h_minus (H/2) + h_plus (H) = 1.5H
            self.odd_mlp  = build_mlp_antisym(config, in_size=H + h_half, hidden_size=H, out_size=2, norm_layer=False)
        def forward(self, graph):
            h_plus, h_minus = graph.edge_attr, graph.edge_attr_asym      # (E,H), (E,H/2)
            even_feats = torch.cat([h_plus, h_minus**2], dim=-1)
            odd_feats  = torch.cat([h_minus, h_plus], dim=-1)
            uvp_qmag = self.even_mlp(even_feats)          # (E,5)
            uv_face = uvp_qmag[:, 0:2]
            p_face  = uvp_qmag[:, 2:3]
            q_mag   = torch.nn.functional.softplus(uvp_qmag[:, 3:5])
            s_odd   = torch.tanh(self.odd_mlp(odd_feats))
            q_n     = q_mag * s_odd
            return torch.cat([uv_face, p_face, q_n], dim=-1)
