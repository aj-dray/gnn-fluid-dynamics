# === LIBRARIES ===


from os import stat
import torch
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


# === FUNCTIONS ===



# === CLASSES ===


class FvgnA(Model):
    """
    Finite Volume Graph Network (FVGN) implementation.
    Based on the original FVGN architecture for fluid dynamics simulation.
    Uses encoder-processor-decoder with physics-informed integration.
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

        # Integrator
        self.integrator = self.Integrator(config, rho=1)

    @classmethod
    def get_feature_sizes(cls, dataset):
        return ([2, 5 + len(dataset.class_types), 0], [0, 5, 0])

    @classmethod
    def get_normalisation_map(cls):
        registry = {
            "cell_velocity_x": (lambda graphs: graphs[0].x[:, 0:1], 'z_score'),
            "cell_velocity_y": (lambda graphs: graphs[0].x[:, 1:2], 'z_score'),
            "cell_velocity_change_x": (lambda graphs: graphs[0].y[:, 0:1], 'z_score'),
            "cell_velocity_change_y": (lambda graphs: graphs[0].y[:, 1:2], 'z_score'),
            "face_velocity_difference_x": (lambda graphs: graphs[1].x[:, 0:1], 'z_score'),
            "face_velocity_difference_y": (lambda graphs: graphs[1].x[:, 1:2], 'z_score'),
            "face_edge_vector_x": (lambda graphs: graphs[1].x[:, 2:3], 'z_score'),
            "face_edge_vector_y": (lambda graphs: graphs[1].x[:, 3:4], 'z_score'),
            "face_area": (lambda graphs: graphs[1].x[:, 4:5], 'z_score'),
            "face_velocity_x": (lambda graphs: graphs[1].y[:, 0:1], 'z_score'),
            "face_velocity_y": (lambda graphs: graphs[1].y[:, 1:2], 'z_score'),
            "face_pressure": (lambda graphs: graphs[1].y[:, 2:3], 'z_score')
        }

        inputs = {
            "cell_velocity_x": (lambda graphs: graphs[0].x[:, 0:1], 'cell_velocity_x'),
            "cell_velocity_y": (lambda graphs: graphs[0].x[:, 1:2], 'cell_velocity_y'),

            "cell_velocity_change_x": (lambda graphs: graphs[0].y[:, 0:1], 'cell_velocity_change_x'),
            "cell_velocity_change_y": (lambda graphs: graphs[0].y[:, 1:2], 'cell_velocity_change_y'),

            "face_velocity_difference_x": (lambda graphs: graphs[1].x[:, 0:1], 'face_velocity_difference_x'),
            "face_velocity_difference_y": (lambda graphs: graphs[1].x[:, 1:2], 'face_velocity_difference_y'),
            "face_edge_vector_x": (lambda graphs: graphs[1].x[:, 2:3], 'face_edge_vector_x'),
            "face_edge_vector_y": (lambda graphs: graphs[1].x[:, 3:4], 'face_edge_vector_y'),
            "face_area": (lambda graphs: graphs[1].x[:, 4:5], 'face_area'),

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
        face_graph.x = torch.cat([face_velocity_change, face_edge_vector, face_graph.area, face_type_one_hot], dim=1)
        face_graph.y = torch.cat([face_graph.velocity[:, -1], face_graph.pressure[:, -1]], dim=1)

        # Clean graphs - i.e. remove unneeded tensors
        graphs = transforms.clean_graphs([cell_graph, face_graph, vertex_graph])

        return graphs

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

        f_graph.x[:, 0:2] = face_velocity_change

        return [c_graph, f_graph, v_graph]

    def forward(self, graphs, mode='rollout'):
        graphs = self.normalizer.input(graphs)
        c_graph, f_graph, v_graph = graphs
        c_graph.edge_attr = f_graph.x
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

    def loss(self, output, graphs):
        c_graph, f_graph, v_graph = graphs
        loss_func = self.loss_func

        face_area = normalize_face_area(f_graph.area, c_graph.volume, c_graph.edge_index, self.dt, self.integrator.face_area_norm)
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
            flux_D = edge_output[:, 3:]

            uu_vu = torch.cat([uv_face[:, 0:1]*uv_face, uv_face[:, 1:2]*uv_face], dim=-1)

            A0 = chain_flux_dot_product(uu_vu[cell_face[0]], unv[:, 0, :]) * e0
            A1 = chain_flux_dot_product(uu_vu[cell_face[1]], unv[:, 1, :]) * e1
            A2 = chain_flux_dot_product(uu_vu[cell_face[2]], unv[:, 2, :]) * e2
            Phi_A = A0 + A1 + A2

            D0 = flux_D[cell_face[0], :]
            D1 = flux_D[cell_face[1], :]
            D2 = flux_D[cell_face[2], :]
            Phi_D = D0 + D1 + D2

            P0 = p_face[cell_face[0]] * unv[:, 0, :] * e0
            P1 = p_face[cell_face[1]] * unv[:, 1, :] * e1
            P2 = p_face[cell_face[2]]* unv[:, 2, :] * e2
            Phi_P = P0 + P1 + P2

            return rhs_coef * (-Phi_A - Phi_P / self.rho) + Phi_D

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
            prev_edge_attr = c_graph.edge_attr.clone()
            prev_node_attr = c_graph.x.clone()

            c_graph = self.cell_block(c_graph, v_graph) # "twice-message passing"
            c_graph = self.face_block(c_graph)

            edge_attr = prev_edge_attr + c_graph.edge_attr # residual connections
            node_attr = prev_node_attr + c_graph.x

            return Data(x=node_attr, edge_attr=edge_attr, edge_index=c_graph.edge_index)

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
            self.face_mlp = build_mlp(config, hidden_size, hidden_size, output_sizes[1], norm_layer=False)

        def forward(self, graph):
            return self.face_mlp(graph.edge_attr)


class FvgnB(FvgnA):
    """
    Real space integration
    """
    face_grad_weights_use = True

    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)
        self.integrator = self.Integrator(config, rho=1, nu=1e-3)
        self.face_mls_weights = MovingLeastSquaresWeights(config, loc='face')

    @classmethod
    def get_feature_sizes(cls, dataset):
        return ([2, 5 + len(dataset.class_types), 0], [0, 3, 0]) # no longer need to predict diffusion

    @classmethod
    def get_normalisation_map(cls):
        registry, inputs, outputs = super().get_normalisation_map()
        # Remove diffusion from normalisation map
        registry.update(
            {"face_area": (lambda graphs: graphs[1].x[:, 4:5], 'z_score'),}
        )
        return registry, inputs, outputs

    def forward(self, graphs, mode='rollout'):
        graphs = self.normalizer.input(graphs)
        c_graph, f_graph, v_graph = graphs
        c_graph.edge_attr = f_graph.x
        c_graph_geom = c_graph.clone()

        c_graph = self.encoder(c_graph)

        for gnblock in self.processer_list:
            c_graph = gnblock(c_graph, v_graph)

        edge_attr_out = self.decoder(c_graph) # ouputs normalised

        output = self.normalizer.output([None, edge_attr_out.clone(), None], inverse=True) # denormalise for integrator
        self.dt = c_graph_geom.dt
        acc_pred = self.integrator(output[1], c_graph_geom, f_graph, self.dt) # outputs denormalised

        output = [acc_pred, output[1], None]
        if mode == 'train':
            output = self.normalizer.output(output)

        return {
            'cell_velocity_change': output[0],
            'face_velocity': output[1][:, :2],
            'face_pressure': output[1][:, 2:3],
        }

    def loss(self, output, graphs):
        c_graph, f_graph, v_graph = graphs
        loss_func = self.loss_func

        face_area = f_graph.x[:, 4:5] # normalised face_area
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

    class Integrator(torch.nn.Module):
        def __init__(self, config, rho, nu=None):
            super().__init__()
            self.rho = rho
            self.nu = nu

        def forward(self, edge_output, c_graph, f_graph, dt):
            unv = c_graph.normal
            face_area = f_graph.area # not normalised in forward pass
            cell_face = f_graph.face

            e0 = face_area[cell_face[0]]
            e1 = face_area[cell_face[1]]
            e2 = face_area[cell_face[2]]

            uv_face = edge_output[:, :2]
            p_face = edge_output[:, 2:3]

            uu_vu = torch.cat([uv_face[:, 0:1]*uv_face, uv_face[:, 1:2]*uv_face], dim=-1)
            A0 = chain_flux_dot_product(uu_vu[cell_face[0]], unv[:, 0, :]) * e0
            A1 = chain_flux_dot_product(uu_vu[cell_face[1]], unv[:, 1, :]) * e1
            A2 = chain_flux_dot_product(uu_vu[cell_face[2]], unv[:, 2, :]) * e2
            Phi_A = A0 + A1 + A2

            velocity_gradient = geometry.calc_gradient_tensor(uv_face, f_graph.grad_weights, f_graph.grad_neighbours)
            A0 = chain_flux_dot_product(velocity_gradient[cell_face[0]], unv[:, 0, :]) * e0
            A1 = chain_flux_dot_product(velocity_gradient[cell_face[1]], unv[:, 1, :]) * e1
            A2 = chain_flux_dot_product(velocity_gradient[cell_face[2]], unv[:, 2, :]) * e2
            Phi_D = A0 + A1 + A2

            P0 = p_face[cell_face[0]] * unv[:, 0, :] * e0
            P1 = p_face[cell_face[1]] * unv[:, 1, :] * e1
            P2 = p_face[cell_face[2]]* unv[:, 2, :] * e2
            Phi_P = P0 + P1 + P2

            return torch.mean(dt) / c_graph.volume * (-Phi_A - Phi_P / self.rho + self.nu * Phi_D)


class FvgnC(FvgnA):
    """Use Temporal Bundling"""
    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)

        # Adjust for temporal bundling
        window_size = config.model.bundle_size
        self.output_sizes = [output * window_size for output in self.output_sizes]
        self.decoder = self.Decoder(config, self.hidden_size, self.output_sizes)

        # Integrator
        self.integrator = self.Integrator(config, rho=1)

    @classmethod
    def transform_features(cls, dataset, graphs, mesh_id=None):
        cell_graph, face_graph, vertex_graph = graphs
        cell_velocity = cell_graph.velocity[:, 0:1]
        if dataset.noise:
            cell_velocity = transforms.add_noise(cell_velocity, std=dataset.config.training.noise_std)
        cell_graph.x = cell_velocity.flatten(start_dim=1)
        cell_target = cell_graph.velocity[:, 1:]
        cell_graph.y = cell_target - cell_velocity

        # Flip edges
        if dataset.mode == 'train':
            cell_graph.edge_index, flip_mask = transforms.random_edge_flip(cell_graph.edge_index)
            safe_flip = flip_mask & (cell_graph.edge_index[0] != cell_graph.edge_index[1]) # don't flip on boundaries
            face_graph.normal[safe_flip] *= -1

        # Use FVGN-like boundaries
        face_type = face_graph.type
        f_interior_mask = (face_type == dataset.class_types.NORMAL) | (face_type == dataset.class_types.OUTFLOW) | (face_type == dataset.class_types.SLIP) | (face_type == dataset.class_types.WALL_BOUNDARY)
        face_graph.boundary_mask = ~f_interior_mask.squeeze()


        face_velocity_change = transforms.calc_face_velocity_change(cell_velocity.squeeze(1), cell_graph.edge_index)
        face_velocity_change[face_graph.boundary_mask] = face_graph.velocity[:, 0][face_graph.boundary_mask] # apply boundary conditions manually
        face_edge_vector  = transforms.calc_cell_edge_vector(cell_graph.pos, cell_graph.edge_index)
        face_type_one_hot = transforms.calc_face_type_one_hot(face_graph.type, len(dataset.class_types))
        face_graph.x = torch.cat([face_velocity_change, face_edge_vector, face_graph.area, face_type_one_hot], dim=1)
        face_target = torch.cat([face_graph.velocity[:, 1:], face_graph.pressure[:, 1:]], dim=2)
        face_graph.y = face_target

        graphs = transforms.clean_graphs([cell_graph, face_graph, vertex_graph])
        return graphs

    @classmethod
    def get_normalisation_map(cls):
        registry = {
            "cell_velocity_x": (lambda graphs: graphs[0].x[:, 0:1], 'z_score'),
            "cell_velocity_y": (lambda graphs: graphs[0].x[:, 1:2], 'z_score'),
            "cell_velocity_change_x": (lambda graphs: graphs[0].y[:, :, 0:1], 'z_score'),
            "cell_velocity_change_y": (lambda graphs: graphs[0].y[:, :, 1:2], 'z_score'),
            "face_velocity_difference_x": (lambda graphs: graphs[1].x[:, 0:1], 'z_score'),
            "face_velocity_difference_y": (lambda graphs: graphs[1].x[:, 1:2], 'z_score'),
            "face_edge_vector_x": (lambda graphs: graphs[1].x[:, 2:3], 'z_score'),
            "face_edge_vector_y": (lambda graphs: graphs[1].x[:, 3:4], 'z_score'),
            "face_area": (lambda graphs: graphs[1].x[:, 4:5], 'z_score'),
            "face_velocity_x": (lambda graphs: graphs[1].y[:, 0, 0:1], 'z_score'),
            "face_velocity_y": (lambda graphs: graphs[1].y[:, 0, 1:2], 'z_score'),
            "face_pressure": (lambda graphs: graphs[1].y[:, 0, 2:3], 'z_score'),
        }

        inputs = {
            "cell_velocity_x": (lambda graphs: graphs[0].x[:, 0:1], 'cell_velocity_x'),
            "cell_velocity_y": (lambda graphs: graphs[0].x[:, 1:2], 'cell_velocity_y'),

            "cell_velocity_change_x": (lambda graphs: graphs[0].y[:, :, 0:1], 'cell_velocity_change_x'),
            "cell_velocity_change_y": (lambda graphs: graphs[0].y[:, :, 1:2], 'cell_velocity_change_y'),

            "face_velocity_difference_x": (lambda graphs: graphs[1].x[:, 0:1], 'face_velocity_difference_x'),
            "face_velocity_difference_y": (lambda graphs: graphs[1].x[:, 1:2], 'face_velocity_difference_y'),
            "face_edge_vector_x": (lambda graphs: graphs[1].x[:, 2:3], 'face_edge_vector_x'),
            "face_edge_vector_y": (lambda graphs: graphs[1].x[:, 3:4], 'face_edge_vector_y'),
            "face_area": (lambda graphs: graphs[1].x[:, 4:5], 'face_area'),

            "face_velocity_x": (lambda graphs: graphs[1].y[:, :, 0:1], 'face_velocity_x'),
            "face_velocity_y": (lambda graphs: graphs[1].y[:, :, 1:2], 'face_velocity_y'),
            "face_pressure": (lambda graphs: graphs[1].y[:, :, 2:3], 'face_pressure')
        }

        outputs = {
            "cell_velocity_change_x": (lambda outputs: outputs[0][:, :, 0:1], 'cell_velocity_change_x'),
            "cell_velocity_change_y": (lambda outputs: outputs[0][:, :, 1:2], 'cell_velocity_change_y'),

            "face_velocity_x": (lambda outputs: outputs[1][:, :, 0:1], 'face_velocity_x'),
            "face_velocity_y": (lambda outputs: outputs[1][:, :, 1:2], 'face_velocity_y'),
            "face_pressure": (lambda outputs: outputs[1][:, :, 2:3], 'face_pressure')
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
        face_velocity_change[mask] = f_graph.y[:, -1, 0:2][mask] # apply boundary conditions manually

        f_graph.x[:, 0:2] = face_velocity_change

        return [c_graph, f_graph, v_graph]

    def forward(self, graphs, mode='rollout'):
        graphs = self.normalizer.input(graphs)
        c_graph, f_graph, v_graph = graphs
        c_graph.edge_attr = f_graph.x
        c_graph_geom = c_graph.clone()

        c_graph = self.encoder(c_graph)

        for gnblock in self.processer_list:
            c_graph = gnblock(c_graph, v_graph)

        edge_attr_out = self.decoder(c_graph)

        self.dt = c_graph_geom.dt
        acc_pred = self.integrator(edge_attr_out, c_graph_geom, f_graph, self.dt)
        output = [acc_pred, edge_attr_out, None]
        # print(acc_pred.shape)
        if mode == 'rollout':
            output = self.normalizer.output(output, inverse=True)

        return {
            'cell_velocity_change': output[0][:, :, 0:2],
            'face_velocity':  output[1][:, :, 0:2],
            'face_pressure': output[1][:, :, 2:3],
        }

    def loss(self, output, graphs):
        c_graph, f_graph, v_graph = graphs
        loss_func = self.loss_func

        # Get number of timesteps
        k = output['face_velocity'].shape[1]

        total_losses = []
        continuity_losses = []
        cell_velocity_change_losses = []
        face_velocity_losses = []
        face_pressure_losses = []

        for t in range(k):
            face_area = normalize_face_area(f_graph.area, c_graph.volume, c_graph.edge_index, self.dt, self.integrator.face_area_norm)
            cell_divergence = fvm.divergence_from_uf(output['face_velocity'][:, t, :], c_graph.normal, face_area, f_graph.face)
            continuity = loss_func(cell_divergence,
                                            torch.zeros_like(cell_divergence),
                                            None,
                                            c_graph.batch)

            cell_velocity_change = loss_func(output['cell_velocity_change'][:, t, :],
                                            c_graph.y[:, t, :],
                                            None,
                                            c_graph.batch)

            face_velocity = loss_func(output['face_velocity'][:, t, :],
                                            f_graph.y[:, t, :2],
                                            ~f_graph.boundary_mask, # only interior
                                            f_graph.batch)

            face_pressure = loss_func(output['face_pressure'][:, t, :],
                                            f_graph.y[:, t, 2:3],
                                            None,
                                            f_graph.batch)

            w = self.config.training.loss_weights
            total = w['continuity'] * continuity + w['cell_velocity_change'] * cell_velocity_change + w['face_velocity'] * face_velocity + w['face_pressure'] * face_pressure

            total_losses.append(total)
            continuity_losses.append(continuity)
            cell_velocity_change_losses.append(cell_velocity_change)
            face_velocity_losses.append(face_velocity)
            face_pressure_losses.append(face_pressure)

        # Average over timesteps
        total_loss = torch.mean(torch.stack(total_losses))
        loss = torch.mean(torch.log(total_loss))

        return {
            "total_log_loss": loss,
            "continuity_loss": torch.mean(torch.stack(continuity_losses)),
            "cell_velocity_change_loss": torch.mean(torch.stack(cell_velocity_change_losses)),
            "face_velocity_loss": torch.mean(torch.stack(face_velocity_losses)),
            "face_pressure_loss": torch.mean(torch.stack(face_pressure_losses))
        }

    class Integrator(torch.nn.Module):
        def __init__(self, config, rho):
            super().__init__()
            self.rho = rho
            self.face_area_norm = torch.nn.BatchNorm1d(1)
            self.face_area = None

        def forward(self, edge_output, c_graph, f_graph, dt):
            # print("edge output ", edge_output.shape)
            unv = c_graph.normal
            rhs_coef = 1.0
            face_area = f_graph.area
            cell_face = f_graph.face
            k = edge_output.shape[1]

            face_area = normalize_face_area(face_area, c_graph.volume, c_graph.edge_index, dt, self.face_area_norm)
            self.face_area = face_area

            e0 = face_area[cell_face[0]]
            e1 = face_area[cell_face[1]]
            e2 = face_area[cell_face[2]]

            results = []
            for t in range(k):
                uv_face = edge_output[:, t, :2]
                p_face = edge_output[:, t, 2:3]
                flux_D = edge_output[:, t, 3:]

                uu_vu = torch.cat([uv_face[:, 0:1]*uv_face, uv_face[:, 1:2]*uv_face], dim=-1)

                A0 = chain_flux_dot_product(uu_vu[cell_face[0]], unv[:, 0, :]) * e0
                A1 = chain_flux_dot_product(uu_vu[cell_face[1]], unv[:, 1, :]) * e1
                A2 = chain_flux_dot_product(uu_vu[cell_face[2]], unv[:, 2, :]) * e2
                Phi_A = A0 + A1 + A2

                D0 = flux_D[cell_face[0], :]
                D1 = flux_D[cell_face[1], :]
                D2 = flux_D[cell_face[2], :]
                Phi_D = D0 + D1 + D2

                P0 = p_face[cell_face[0]] * unv[:, 0, :] * e0
                P1 = p_face[cell_face[1]] * unv[:, 1, :] * e1
                P2 = p_face[cell_face[2]]* unv[:, 2, :] * e2
                Phi_P = P0 + P1 + P2

                result = rhs_coef * (-Phi_A - Phi_P / self.rho) + Phi_D
                results.append(result * (k+1))

            return torch.stack(results, dim=1)

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
            prev_edge_attr = c_graph.edge_attr.clone()
            prev_node_attr = c_graph.x.clone()

            c_graph = self.cell_block(c_graph, v_graph) # "twice-message passing"
            c_graph = self.face_block(c_graph)

            edge_attr = prev_edge_attr + c_graph.edge_attr # residual connections
            node_attr = prev_node_attr + c_graph.x

            return Data(x=node_attr, edge_attr=edge_attr, edge_index=c_graph.edge_index)

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
            self.face_mlp = build_mlp(config, hidden_size, hidden_size, output_sizes[1], norm_layer=False)

        def forward(self, graph):
            # graph.edge_attr: (a, b*k), want (a, k, b)
            out = self.face_mlp(graph.edge_attr)
            a, bk = out.shape
            b = 5
            k = bk // b
            return out.view(a, k, b)


class FvgnD(FvgnA):
    """Push forward trick. Change target to be final acceleration rather than total change over window"""
    pushforward_use = True
    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)
        self.pushforward_use = True

    @classmethod
    def transform_features(cls, dataset, graphs, mesh_id=None):
        cell_graph, face_graph, vertex_graph = graphs
        cell_velocity = cell_graph.velocity[:, 0]
        if dataset.noise:
            cell_velocity = transforms.add_noise(cell_velocity, std=dataset.config.training.noise_std)
        cell_graph.x = cell_velocity
        cell_graph.y = cell_graph.velocity[:, -1] # final change is target

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
        face_graph.y = torch.cat([face_graph.velocity[:, -1], face_graph.pressure[:, -1]], dim=1)

        # Clean graphs - i.e. remove unneeded tensors
        # graphs = transforms.clean_graphs([cell_graph, face_graph, vertex_graph])
        return graphs

    @classmethod
    def get_normalisation_map(cls):
        registry, inputs, outputs = super().get_normalisation_map()

        registry.update({
            "cell_velocity_change_x": (lambda graphs: graphs[0].velocity[:, -1, 0:1] - graphs[0].velocity[:, -2, 0:1], 'z_score'),
            "cell_velocity_change_y": (lambda graphs: graphs[0].velocity[:, -1, 1:2] - graphs[0].velocity[:, -2, 1:2], 'z_score'),
        })

        return registry, inputs, outputs


class FvgnE(FvgnA):
    """Physical Normalisation"""
    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)
        self.pushfoward_use = False

    @classmethod
    def get_normalisation_map(cls):
        registry = {
            "characteristic_velocity": (lambda graphs: torch.norm(graphs[0].x[:, 0:2], dim=1), 'max_scale'),
            "characteristic_length": (lambda graphs: torch.sqrt(graphs[0].volume), 'mean_scale'),
            "characteristic_pressure": (None, 'max_scale'), # custom implemented
        }

        inputs = {
            "cell_velocity_x": (lambda graphs: graphs[0].x[:, 0:1], 'characteristic_velocity'),
            "cell_velocity_y": (lambda graphs: graphs[0].x[:, 1:2], 'characteristic_velocity'),

            "cell_velocity_change_x": (lambda graphs: graphs[0].y[:, 0:1], 'characteristic_velocity'),
            "cell_velocity_change_y": (lambda graphs: graphs[0].y[:, 1:2], 'characteristic_velocity'),

            "face_velocity_difference_x": (lambda graphs: graphs[1].x[:, 0:1], 'characteristic_velocity'),
            "face_velocity_difference_y": (lambda graphs: graphs[1].x[:, 1:2], 'characteristic_velocity'),
            "face_edge_vector_x": (lambda graphs: graphs[1].x[:, 2:3], 'characteristic_length'),
            "face_edge_vector_y": (lambda graphs: graphs[1].x[:, 3:4], 'characteristic_length'),
            "face_area": (lambda graphs: graphs[1].x[:, 4:5], 'characteristic_length'),

            "face_velocity_x": (lambda graphs: graphs[1].y[:, 0:1], 'characteristic_velocity'),
            "face_velocity_y": (lambda graphs: graphs[1].y[:, 1:2], 'characteristic_velocity'),
            "face_pressure": (lambda graphs: graphs[1].y[:, 2:3], 'characteristic_pressure')
        }

        outputs = {
            "cell_velocity_change_x": (lambda outputs: outputs[0][:, 0:1], 'characteristic_velocity'),
            "cell_velocity_change_y": (lambda outputs: outputs[0][:, 1:2], 'characteristic_velocity'),

            "face_velocity_x": (lambda outputs: outputs[1][:, 0:1], 'characteristic_velocity'),
            "face_velocity_y": (lambda outputs: outputs[1][:, 1:2], 'characteristic_velocity'),
            "face_pressure": (lambda outputs: outputs[1][:, 2:3], 'characteristic_pressure')
        }

        return registry, inputs, outputs


class FvgnF(FvgnA):
    """Single GNBlock Weights"""
    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)

        # Encoder-Processor-Decoder
        self.encoder = self.Encoder(config, self.input_sizes, self.hidden_size)
        # Single GN block instead of multiple
        self.gn_block = self.GN_Block(config, self.hidden_size)
        self.decoder = self.Decoder(config, self.hidden_size, self.output_sizes)
        self.mp_num = config.model.mp_num

    def forward(self, graphs, mode='rollout'):
        graphs = self.normalizer.input(graphs)
        c_graph, f_graph, v_graph = graphs
        c_graph.edge_attr = f_graph.x
        c_graph_geom = c_graph.clone()

        c_graph = self.encoder(c_graph)

        # Use single GN block mp_num times with step parameter
        for mp_step in range(self.mp_num):
            step_param = (mp_step + 1) / self.mp_num  # Normalized step from 1/mp_num to 1.0
            c_graph = self.gn_block(c_graph, v_graph, step_param)

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

        def forward(self, c_graph, v_graph, step):
            prev_edge_attr = c_graph.edge_attr.clone()
            prev_node_attr = c_graph.x.clone()

            c_graph = self.cell_block(c_graph, v_graph, step) # "twice-message passing"
            c_graph = self.face_block(c_graph, step)

            edge_attr = prev_edge_attr + c_graph.edge_attr # residual connections
            node_attr = prev_node_attr + c_graph.x

            return Data(x=node_attr, edge_attr=edge_attr, edge_index=c_graph.edge_index)

        class Face_Block(torch.nn.Module):
            def __init__(self, config, hidden_size):
                super().__init__()
                input_size = hidden_size * 3 + 1  # edge + two cells + step
                self.face_mlp = build_mlp(config, input_size, hidden_size, out_size=hidden_size)

            def forward(self, c_graph, step):
                row, col = c_graph.edge_index
                # Create step tensor with same device and dtype as edge_attr
                step_tensor = torch.full((c_graph.edge_attr.size(0), 1), step,
                                       device=c_graph.edge_attr.device,
                                       dtype=c_graph.edge_attr.dtype)

                aggr_features = torch.cat([c_graph.edge_attr, c_graph.x[row], c_graph.x[col], step_tensor], dim=1)
                edge_attr = self.face_mlp(aggr_features)
                return Data(x=c_graph.x, edge_attr=edge_attr, edge_index=c_graph.edge_index)

        class Cell_Block(torch.nn.Module):
            def __init__(self, config, hidden_size, mp_times=2):
                super().__init__()
                input_size = hidden_size + hidden_size // 2 + 1  # Added +1 for step
                self.cell_mlp = build_mlp(config, input_size, hidden_size, out_size=hidden_size)
                self.mp_times = mp_times

            def forward(self, c_graph, v_graph, step):
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

                # Create step tensor with same device and dtype as collected_features
                step_tensor = torch.full((collected_features.size(0), 1), step,
                                       device=collected_features.device,
                                       dtype=collected_features.dtype)

                cell_attr = self.cell_mlp(torch.cat([collected_features, step_tensor], dim=-1))

                return Data(x=cell_attr, edge_attr=edge_attr, edge_index=c_graph.edge_index)

    class Decoder(torch.nn.Module):
        def __init__(self, config, hidden_size, output_sizes):
            super().__init__()
            self.face_mlp = build_mlp(config, hidden_size, hidden_size, output_sizes[1], norm_layer=False)

        def forward(self, graph):
            return self.face_mlp(graph.edge_attr)


class FvgnH(FvgnA):
    """Augment input features"""
    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)

    @classmethod
    def get_feature_sizes(cls, dataset):
        return ([2, 7 + len(dataset.class_types), 0], [0, 5, 0])

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
        face_edge_distance = torch.norm(face_edge_vector, dim=1, keepdim=True)
        small_distance_mask = face_edge_distance < 1e-8

        # For normal cases, compute angle as before
        face_edge_vector_norm = face_edge_vector / (face_edge_distance + 1e-8)
        dot_product = (face_edge_vector_norm * face_graph.normal).sum(dim=1, keepdim=True)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)  # Clamp to valid range for acos
        angle = torch.acos(torch.abs(dot_product))  # Take absolute value to get minimum angle
        angle = torch.where(small_distance_mask, torch.zeros_like(angle), angle)
        # nan_count = torch.isnan(angle).sum()
        # print(f"Number of NaN values in angle: {nan_count}")

        face_graph.x = torch.cat([face_velocity_change, face_graph.normal, face_graph.area, face_edge_distance, angle, face_type_one_hot], dim=1)
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
            "face_normal_x": (lambda graphs: graphs[1].x[:, 2:3], 'z_score'),
            "face_normal_y": (lambda graphs: graphs[1].x[:, 3:4], 'z_score'),
            "face_area": (lambda graphs: graphs[1].x[:, 4:5], 'z_score'),
            "face_adjacent_distance" : (lambda graphs: graphs[1].x[:, 5:6], 'z_score'),
            "face_angle": (lambda graphs: graphs[1].x[:, 6:7], 'z_score'),
            "face_velocity_x": (lambda graphs: graphs[1].y[:, 0:1], 'z_score'),
            "face_velocity_y": (lambda graphs: graphs[1].y[:, 1:2], 'z_score'),
            "face_pressure": (lambda graphs: graphs[1].y[:, 2:3], 'z_score'),
            "face_velocity_difference_x": (lambda graphs: graphs[1].x[:, 0:1], 'z_score'),
            "face_velocity_difference_y": (lambda graphs: graphs[1].x[:, 1:2], 'z_score'),
        }

        inputs = {
            # inputs
            "cell_velocity_x": (lambda graphs: graphs[0].x[:, 0:1], 'cell_velocity_x'),
            "cell_velocity_y": (lambda graphs: graphs[0].x[:, 1:2], 'cell_velocity_y'),
            "face_velocity_difference_x": (lambda graphs: graphs[1].x[:, 0:1], 'face_velocity_difference_x'),
            "face_velocity_difference_y": (lambda graphs: graphs[1].x[:, 1:2], 'face_velocity_difference_y'),
            "face_area": (lambda graphs: graphs[1].x[:, 4:5], 'face_area'),
            "face_adjacent_distance": (lambda graphs: graphs[1].x[:, 5:6], 'face_adjacent_distance'),
            "face_angle": (lambda graphs: graphs[1].x[:, 6:7], 'face_angle'),
            "face_normal_x": (lambda graphs: graphs[1].x[:, 2:3], 'face_normal_x'),
            "face_normal_y": (lambda graphs: graphs[1].x[:, 3:4], 'face_normal_y'),
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


class FvgnI(FvgnA):
    """Turn off wall B.C.s"""
    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)

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

        f_graph.x[:, 0:2] = face_velocity_change

        return [c_graph, f_graph, v_graph]


class FvgnJ(FvgnA):
    """Learned real spacing scaling"""
    face_grad_weights_use = False

    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)
        self.integrator = self.Integrator(config, rho=1, nu=1e-3)
        self.face_mls_weights = MovingLeastSquaresWeights(config, loc='face')

        self.velocity_scale_x = torch.nn.Parameter(torch.tensor(1.0))
        self.velocity_scale_y = torch.nn.Parameter(torch.tensor(0.01))
        self.pressure_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.diffusion_scale = torch.nn.Parameter(torch.tensor(1.0))

        self.velocity_bias_x = torch.nn.Parameter(torch.tensor(0.0))
        self.velocity_bias_y = torch.nn.Parameter(torch.tensor(0.0))
        self.pressure_bias = torch.nn.Parameter(torch.tensor(0.0))
        self.diffusion_bias = torch.nn.Parameter(torch.tensor(0.0))


    @classmethod
    def get_feature_sizes(cls, dataset):
        return ([2, 5 + len(dataset.class_types), 0], [0, 5, 0]) # no longer need to predict diffusion

    def forward(self, graphs, mode='rollout'):
        graphs = self.normalizer.input(graphs)
        c_graph, f_graph, v_graph = graphs
        c_graph.edge_attr = f_graph.x
        c_graph_geom = c_graph.clone()

        c_graph = self.encoder(c_graph)

        for gnblock in self.processer_list:
            c_graph = gnblock(c_graph, v_graph)

        edge_attr_out_raw = self.decoder(c_graph) # ouputs normalised

        u_face = edge_attr_out_raw[:, 0:1] * self.velocity_scale_x + self.velocity_bias_x
        v_face = edge_attr_out_raw[:, 1:2] * self.velocity_scale_y + self.velocity_bias_y
        uv_face = torch.cat([u_face, v_face], dim=-1)
        p_face = edge_attr_out_raw[:, 2:3] * self.pressure_scale + self.pressure_bias
        d_flux = edge_attr_out_raw[:, 3:5] * self.diffusion_scale + self.diffusion_bias

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

    def loss(self, output, graphs):
        c_graph, f_graph, v_graph = graphs
        loss_func = self.loss_func

        face_area = f_graph.x[:, 4:5] # normalised face_area
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

    class Integrator(torch.nn.Module):
        def __init__(self, config, rho, nu=None):
            super().__init__()
            self.rho = rho
            self.nu = nu

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

            D0 = q_face[cell_face[0], :]
            D1 = q_face[cell_face[1], :]
            D2 = q_face[cell_face[2], :]
            Phi_D = (D0 + D1 + D2)

            P0 = p_face[cell_face[0]] * unv[:, 0, :] * e0
            P1 = p_face[cell_face[1]] * unv[:, 1, :] * e1
            P2 = p_face[cell_face[2]]* unv[:, 2, :] * e2
            Phi_P = P0 + P1 + P2

            return torch.mean(dt) / c_graph.volume * (-Phi_A - Phi_P / self.rho + self.nu * Phi_D )


class FvgnK(FvgnA):
    """Learned real spacing dimensionless"""
    # face_grad_weights_use = True

    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)
        self.integrator = self.Integrator(config, rho=1, nu=1e-3)
        self.face_mls_weights = MovingLeastSquaresWeights(config, loc='face')
        self.anisotropy_ratio = torch.nn.Parameter(torch.tensor(0.0001))

    @classmethod
    def get_feature_sizes(cls, dataset):
        return ([2, 5 + len(dataset.class_types), 0], [0, 5, 0]) # no longer need to predict diffusion

    def forward(self, graphs, mode='rollout'):
        mask = graphs[1].type == NodeType.INFLOW
        u_ref = []
        for batch_idx in torch.unique(graphs[0].batch):
            batch_mask = (graphs[1].batch == batch_idx) & mask
            if batch_mask.any():
                batch_data = graphs[1].y[batch_mask]
                u_ref.append(batch_data[0, 0])
            else:
                u_ref.append(torch.tensor(1.0, device=graphs[0].y.device))
        u_ref = torch.stack(u_ref)
        Re = graphs[0].Re
        l_ref = Re * 1e-3 / u_ref
        u_ref = u_ref[graphs[1].batch].unsqueeze(-1)
        l_ref = l_ref[graphs[1].batch].unsqueeze(-1)
        p_ref = u_ref**2  # (rho = 1)
        d_ref = u_ref * l_ref

        graphs = self.normalizer.input(graphs)
        c_graph, f_graph, v_graph = graphs
        c_graph.edge_attr = f_graph.x
        c_graph_geom = c_graph.clone()

        c_graph = self.encoder(c_graph)

        for gnblock in self.processer_list:
            c_graph = gnblock(c_graph, v_graph)

        edge_attr_out = self.decoder(c_graph) # ouputs normalised

        # Into physical space
        edge_attr_out = torch.cat([
            edge_attr_out[:, 0:1] * u_ref,
            edge_attr_out[:, 1:2] * u_ref * self.anisotropy_ratio,
            edge_attr_out[:, 2:3] * p_ref,
            edge_attr_out[:, 3:5] * d_ref
        ], dim=-1)

        self.dt = c_graph_geom.dt.clone()
        acc_pred = self.integrator(edge_attr_out, c_graph_geom, f_graph, self.dt) # outputs denormalised

        output = [acc_pred, edge_attr_out, None]

        if mode != 'rollout': # normalised for training
            output = self.normalizer.output(output) # normalise for loss

        return {
            'cell_velocity_change': output[0][:, 0:2],
            'face_velocity': output[1][:, :2],
            'face_pressure': output[1][:, 2:3],
        }

    def loss(self, output, graphs):
        c_graph, f_graph, v_graph = graphs
        loss_func = self.loss_func

        face_area = f_graph.x[:, 4:5] # normalised face_area
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

    class Integrator(torch.nn.Module):
        def __init__(self, config, rho, nu=None):
            super().__init__()
            self.rho = rho
            self.nu = nu

        def forward(self, edge_output, c_graph, f_graph, dt):

            unv = c_graph.normal
            face_area = f_graph.area
            cell_face = f_graph.face

            e0 = face_area[cell_face[0]]
            e1 = face_area[cell_face[1]]
            e2 = face_area[cell_face[2]]

            uv_face = edge_output[:, 0:2].clone()
            p_face = edge_output[:, 2:3].clone()
            d_flux = edge_output[:, 3:4].clone()

            uu_vu = torch.cat([uv_face[:, 0:1]*uv_face, uv_face[:, 1:2]*uv_face], dim=-1)
            A0 = chain_flux_dot_product(uu_vu[cell_face[0]], unv[:, 0, :]) * e0
            A1 = chain_flux_dot_product(uu_vu[cell_face[1]], unv[:, 1, :]) * e1
            A2 = chain_flux_dot_product(uu_vu[cell_face[2]], unv[:, 2, :]) * e2
            Phi_A = A0 + A1 + A2

            D0 = d_flux[cell_face[0], :]
            D1 = d_flux[cell_face[1], :]
            D2 = d_flux[cell_face[2], :]
            Phi_D = (D0 + D1 + D2)

            P0 = p_face[cell_face[0]] * unv[:, 0, :] * e0
            P1 = p_face[cell_face[1]] * unv[:, 1, :] * e1
            P2 = p_face[cell_face[2]]* unv[:, 2, :] * e2
            Phi_P = P0 + P1 + P2

            return torch.mean(dt) / c_graph.volume * (-Phi_A - Phi_P + Phi_D * 1e-3 )# (nu = 1e-3))
