# === LIBRARIES ===


from os import stat
import torch
from torch_geometric.data import Data
from torch_scatter import scatter_add
import json


# === MODULES ===


from utils.maths import chain_dot_product, chain_flux_dot_product, MovingLeastSquaresWeights
from utils.normalisation import CustomNormalizer, normalize_face_area
import utils.transforms as transforms
import utils.fvm as fvm
from models.Model import Model


# === FUNCTIONS ===


def build_mlp(config, in_size, hidden_size, out_size, norm_layer=True):
    module = torch.nn.Sequential(
        torch.nn.Linear(in_size, hidden_size),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden_size, out_size)
    )
    if norm_layer:
        return torch.nn.Sequential(module, torch.nn.LayerNorm(normalized_shape=out_size))
    return module


# === CLASSES ===


class MgnA(Model):
    """
    MeshGraphNet (MGN) implementation for fluid dynamics.
    Graph neural network that learns mesh-based physics simulations.
    Uses gradient weights for divergence loss computation.
    """
    cell_grad_weights_use = True  # for divergence loss in rollout
    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)

        self.encoder = self.Encoder(config, self.input_sizes, self.hidden_size)
        processer_list = []
        for _ in range(config.model.mp_num):
            processer_list.append(self.GN_Block(config, self.hidden_size))
        self.processer_list = torch.nn.ModuleList(processer_list)
        self.decoder = self.Decoder(config, self.hidden_size, self.output_sizes)

        self.cell_mls_weights = MovingLeastSquaresWeights(config, loc='cell')

    @classmethod
    def get_feature_sizes(cls, dataset):
        return ([2, 5 + len(dataset.class_types), 0], [3, 0, 0])

    @classmethod
    def transform_features(cls, dataset, graphs, mesh_id=None):
        cell_graph, face_graph, vertex_graph = graphs
        cell_velocity = cell_graph.velocity[:, 0]
        cell_velocity_target = cell_graph.velocity[:, -1]
        if dataset.noise:
            cell_velocity = transforms.add_noise(cell_velocity, std=dataset.config.training.noise_std)
        cell_graph.x = cell_velocity
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
        face_graph.x = torch.cat([face_velocity_change, face_edge_vector, face_graph.area, face_type_one_hot], dim=1)
        face_graph.y = face_graph.velocity[:, -1] # for boundary conditions

        # Clean graphs - i.e. remove unneeded tensors
        graphs = transforms.clean_graphs([cell_graph, face_graph, vertex_graph])

        return graphs

    @classmethod
    def get_normalisation_map(cls):
        registry = {
            "cell_velocity_x": (lambda graphs: graphs[0].x[:, 0:1], 'z_score'),
            "cell_velocity_y": (lambda graphs: graphs[0].x[:, 1:2], 'z_score'),
            "cell_velocity_change_x": (lambda graphs: graphs[0].y[:, 0:1], 'z_score'),
            "cell_velocity_change_y": (lambda graphs: graphs[0].y[:, 1:2], 'z_score'),
            "cell_pressure": (lambda graphs: graphs[0].y[:, 2:3], 'z_score'),
            "face_velocity_difference_x": (lambda graphs: graphs[1].x[:, 0:1], 'z_score'),
            "face_velocity_difference_y": (lambda graphs: graphs[1].x[:, 1:2], 'z_score'),
            "face_edge_vector_x": (lambda graphs: graphs[1].x[:, 2:3], 'z_score'),
            "face_edge_vector_y": (lambda graphs: graphs[1].x[:, 3:4], 'z_score'),
            "face_area": (lambda graphs: graphs[1].x[:, 4:5], 'z_score'),
        }

        inputs = {
            # inputs
            "cell_velocity_x": (lambda graphs: graphs[0].x[:, 0:1], 'cell_velocity_x'),
            "cell_velocity_y": (lambda graphs: graphs[0].x[:, 1:2], 'cell_velocity_y'),
            "face_velocity_difference_x": (lambda graphs: graphs[1].x[:, 0:1], 'face_velocity_difference_x'),
            "face_velocity_difference_y": (lambda graphs: graphs[1].x[:, 1:2], 'face_velocity_difference_y'),
            "face_edge_vector_x": (lambda graphs: graphs[1].x[:, 2:3], 'face_edge_vector_x'),
            "face_edge_vector_y": (lambda graphs: graphs[1].x[:, 3:4], 'face_edge_vector_y'),
            "face_area": (lambda graphs: graphs[1].x[:, 4:5], 'face_area'),
            # targets
            "cell_velocity_change_x": (lambda graphs: graphs[0].y[:, 0:1], 'cell_velocity_change_x'),
            "cell_velocity_change_y": (lambda graphs: graphs[0].y[:, 1:2], 'cell_velocity_change_y'),
            "cell_pressure": (lambda graphs: graphs[0].y[:, 2:3], 'cell_pressure'),
            # bc
            "face_velocity_x": (lambda graphs: graphs[1].y[:, 0:1], 'cell_velocity_x'), # for B.Cs
            "face_velocity_y": (lambda graphs: graphs[1].y[:, 1:2], 'cell_velocity_y'),
        }

        outputs = {
            # Cell
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
        face_velocity_change[f_graph.boundary_mask] = f_graph.y[:, 0:2][f_graph.boundary_mask] # apply boundary conditions manually

        f_graph.x[:, 0:2] = face_velocity_change

        return [c_graph, f_graph, v_graph]

    def forward(self, graphs, mode='train'):
        graphs = self.normalizer.input(graphs)
        c_graph, f_graph, v_graph = graphs
        c_graph.edge_attr = f_graph.x

        # Encoder-Processor-Decoder
        c_graph = self.encoder(c_graph)
        for gnblock in self.processer_list:
            c_graph = gnblock(c_graph, v_graph)
        cell_output = self.decoder(c_graph)

        output = (cell_output, None, None)

        # Normalisation
        if mode == 'rollout':
            output = self.normalizer.output(output, inverse=True)

        return {
            'cell_velocity_change': output[0][:, 0:2],
            'cell_pressure': output[0][:, 2:3],
        }

    def loss(self, output, graphs):
        c_graph, f_graph, v_graph = graphs
        loss_func = self.loss_func

        cell_velocity_change_loss = loss_func(output['cell_velocity_change'],
                                        c_graph.y[:, 0:2],
                                        None,
                                        c_graph.batch)

        cell_pressure_loss = loss_func(output['cell_pressure'],
                                        c_graph.y[:, 2:3],
                                        None,
                                        f_graph.batch)

        w = self.config.training.loss_weights
        total_loss = w['cell_velocity_change'] * cell_velocity_change_loss + w['cell_pressure'] * cell_pressure_loss
        total_log_loss = torch.mean(torch.log(total_loss))

        return {
            "total_log_loss": total_log_loss,
            "cell_velocity_change_loss": cell_velocity_change_loss,
            "cell_pressure_loss": cell_pressure_loss
        }

    class Encoder(torch.nn.Module):
        def __init__(self, config, input_sizes, hidden_size):
            super().__init__()
            self.face_mlp = build_mlp(config, input_sizes[1], hidden_size, out_size=hidden_size)
            self.cell_mlp = build_mlp(config, input_sizes[0], hidden_size, out_size=hidden_size)

        def forward(self, cell_graph):
            face_attr = self.face_mlp(cell_graph.edge_attr)
            cell_attr = self.cell_mlp(cell_graph.x)
            return Data(x=cell_attr, edge_attr=face_attr, edge_index=cell_graph.edge_index)

    class GN_Block(torch.nn.Module):
        def __init__(self, config, hidden_size):
            super().__init__()
            self.face_block = self.Face_Block(config, hidden_size)
            self.cell_block = self.Cell_Block(config, hidden_size)

        def forward(self, c_graph, v_graph):
            prev_cell_attr = c_graph.x.clone()
            prev_edge_attr = c_graph.edge_attr.clone()

            c_graph = self.face_block(c_graph)
            c_graph = self.cell_block(c_graph, v_graph) # "twice-message passing"

            edge_attr = prev_edge_attr + c_graph.edge_attr
            cell_attr = prev_cell_attr + c_graph.x

            return Data(x=cell_attr, edge_attr=edge_attr, edge_index=c_graph.edge_index)

        class Face_Block(torch.nn.Module):
            def __init__(self, config, hidden_size):
                super().__init__()
                input_size = hidden_size * 3  # 1 * face + 2 * cells
                self.face_mlp = build_mlp(config, input_size, hidden_size, out_size=hidden_size)

            def forward(self, c_graph):
                row, col = c_graph.edge_index
                aggr_features = torch.cat([c_graph.edge_attr, c_graph.x[row], c_graph.x[col]], dim=1) #NOTE: not symmetrical as depends on face orientation
                edge_attr = self.face_mlp(aggr_features)
                return Data(x=c_graph.x, edge_attr=edge_attr, edge_index=c_graph.edge_index)

        class Cell_Block(torch.nn.Module):
            def __init__(self, config, hidden_size, mp_times=2):
                super().__init__()
                input_size = hidden_size + hidden_size // 2
                self.cell_mlp = build_mlp(config, input_size, hidden_size, out_size=hidden_size)
                self.mp_times = mp_times

            def forward(self, c_graph, v_graph):
                # First message passing: edges -> vertices
                senders_node_idx, receivers_node_idx = v_graph.edge_index
                twoway_node_connections = torch.cat([senders_node_idx, receivers_node_idx], dim=0)

                # Aggregate messages at cells
                edge_attr = c_graph.edge_attr
                fwd_attr, rev_attr = torch.chunk(edge_attr, 2, dim=-1)  # (E, F/2) each
                twoway_edge_attr = torch.cat([fwd_attr, rev_attr], dim=0)
                node_agg_received_edges = scatter_add(twoway_edge_attr, twoway_node_connections, dim=0, dim_size=v_graph.num_nodes)

                # Aggregate vertex features to cells (average of 3 vertices per cell)
                cell_agg = (
                    torch.index_select(node_agg_received_edges, 0, v_graph.face[0]) +
                    torch.index_select(node_agg_received_edges, 0, v_graph.face[1]) +
                    torch.index_select(node_agg_received_edges, 0, v_graph.face[2])
                ) / 3.0
                collected_features = torch.cat([c_graph.x, cell_agg], dim=-1)
                cell_attr = self.cell_mlp(collected_features)

                return Data(x=cell_attr, edge_attr=edge_attr, edge_index=c_graph.edge_index)

    class Decoder(torch.nn.Module):
        def __init__(self, config, hidden_size, output_sizes):
            super().__init__()
            self.face_mlp = build_mlp(config, hidden_size, hidden_size, output_sizes[0], norm_layer=False)

        def forward(self, graph):
            return self.face_mlp(graph.x)


class MgnB(MgnA):
    """Direct prediction, normal norm."""

    cell_grad_weights_use = True # for divergence loss in rollout

    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)

    @classmethod
    def transform_features(cls, dataset, graphs, mesh_id=None):
        cell_graph, face_graph, vertex_graph = graphs
        cell_velocity = cell_graph.velocity[:, 0]
        if dataset.noise:
            cell_velocity = transforms.add_noise(cell_velocity, std=dataset.config.training.noise_std)
        cell_graph.x = cell_velocity
        cell_graph.y = torch.cat([cell_graph.velocity[:, -1], cell_graph.pressure[:, -1]], dim=1)

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
        face_graph.x = torch.cat([face_velocity_change, face_edge_vector, face_graph.area, face_type_one_hot], dim=1)
        face_graph.y = face_graph.velocity[:, -1] # for boundary conditions

        # Clean graphs - i.e. remove unneeded tensors
        graphs = transforms.clean_graphs([cell_graph, face_graph, vertex_graph])

        return graphs

    @classmethod
    def get_normalisation_map(cls):
        registry, inputs, outputs = super().get_normalisation_map()

        del inputs["cell_velocity_change_x"] #change targets
        del inputs["cell_velocity_change_y"]
        inputs.update({
            "cell_velocity_x": (lambda graphs: graphs[0].x[:, 0:1], 'cell_velocity_x'),
            "cell_velocity_y": (lambda graphs: graphs[0].x[:, 1:2], 'cell_velocity_y'),
            "cell_velocity_target_x": (lambda graphs: graphs[0].y[:, 0:1], 'cell_velocity_x'),
            "cell_velocity_target_y": (lambda graphs: graphs[0].y[:, 1:2], 'cell_velocity_y')
        })

        del outputs["cell_velocity_change_x"] #change output
        del outputs["cell_velocity_change_y"]
        outputs.update({
            "cell_velocity_x": (lambda outputs: outputs[0][:, 0:1], 'cell_velocity_x'),
            "cell_velocity_y": (lambda outputs: outputs[0][:, 1:2], 'cell_velocity_y')
        })
        return registry, inputs, outputs


    def forward(self, graphs, mode='train'):
        graphs = self.normalizer.input(graphs)
        c_graph, f_graph, v_graph = graphs
        c_graph.edge_attr = f_graph.x

        # Encoder-Processor-Decoder
        c_graph = self.encoder(c_graph)
        for gnblock in self.processer_list:
            c_graph = gnblock(c_graph, v_graph)
        cell_output = self.decoder(c_graph)

        output = (cell_output, None, None)

        # Normalisation
        if mode == 'rollout':
            output = self.normalizer.output(output, inverse=True)

        return {
            'cell_velocity': output[0][:, 0:2],
            'cell_pressure': output[0][:, 2:3],
        }

    def loss(self, output, graphs):
        c_graph, f_graph, v_graph = graphs
        loss_func = self.loss_func

        cell_divergence = fvm.divergence_from_uc(output['cell_velocity'], c_graph.grad_weights, c_graph.grad_neighbours, c_graph.volume)
        continuity   = loss_func(cell_divergence,
                                        torch.zeros_like(cell_divergence),
                                        None,
                                        c_graph.batch)

        cell_velocity_loss = loss_func(output['cell_velocity'],
                                        c_graph.y[:, 0:2],
                                        None,
                                        c_graph.batch)

        cell_pressure_loss = loss_func(output['cell_pressure'],
                                        c_graph.y[:, 2:3],
                                        None,
                                        f_graph.batch)

        w = self.config.training.loss_weights
        total_loss = w['cell_velocity'] * cell_velocity_loss + w['cell_pressure'] * cell_pressure_loss + w['continuity'] * continuity
        total_log_loss = torch.mean(torch.log(total_loss))

        return {
            "total_log_loss": total_log_loss,
            "cell_velocity_loss": cell_velocity_loss,
            "cell_pressure_loss": cell_pressure_loss,
            "continuity_loss": continuity
        }
   

class MgnC(MgnB):
    """Direct prediction, Physics based norm"""

    cell_grad_weights_use = True # for divergence loss in rollout

    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)

    @classmethod
    def get_normalisation_map(cls):
        registry, inputs, outputs = super().get_normalisation_map()

        del inputs["cell_velocity_x"] #change targets
        del inputs["cell_velocity_y"]
        registry.update({
            'cell_velocity_char': (lambda graphs: torch.norm(graphs[0].x[:, 0:2], dim=1), 'mean_scale')
        })

        inputs.update({
            "cell_velocity_x": (lambda graphs: graphs[0].x[:, 0:1], 'cell_velocity_char'),
            "cell_velocity_y": (lambda graphs: graphs[0].x[:, 1:2], 'cell_velocity_char'),
            "cell_velocity_target_x": (lambda graphs: graphs[0].y[:, 0:1], 'cell_velocity_char'),
            "cell_velocity_target_y": (lambda graphs: graphs[0].y[:, 1:2], 'cell_velocity_char')
        })

        outputs.update({
            "cell_velocity_x": (lambda outputs: outputs[0][:, 0:1], 'cell_velocity_char'),
            "cell_velocity_y": (lambda outputs: outputs[0][:, 1:2], 'cell_velocity_char')
        })

        return registry, inputs, outputs
