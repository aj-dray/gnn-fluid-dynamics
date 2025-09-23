# === LIBRARIES ===


from os import stat
import torch
from torch_geometric.data import Data
from torch_scatter import scatter_add


# === MODULES ===


from models.Flux import FluxA, FluxC, FvgnA
from models.Fvgn import build_mlp
from utils.normalisation import normalize_face_area, normalize_vol_dt
from utils.maths import chain_dot_product, chain_flux_dot_product, MovingLeastSquaresWeights
import utils.geometry as geometry
import utils.transforms as transforms
import utils.fvm as fvm


# === FUNCTIONS ===


def calc_cell_flux_from_vertices(vertex_out, graphs):
    """
    Returns:
        (num_cells, 3) as flux for each cell
    """
    c_graph, f_graph, v_graph = graphs
    # Calculate edge differences
    v_vals = vertex_out[v_graph.face]  # Shape: (3, num_cells, 1)
    edge_diffs = torch.stack([
        v_vals[1] - v_vals[2],  # v1 - v2
        v_vals[2] - v_vals[0],  # v2 - v0
        v_vals[0] - v_vals[1]   # v0 - v1
    ], dim=0)  # Shape: (3, num_cells, 1)
    # Squeeze the last dimension and transpose to get (num_cells, 3)
    result = edge_diffs.squeeze(-1).T
    return result



# === CLASSES ===


class VertPotA(FluxA):
    """Predicts u_f, p_f and flux_f. No loss on flux_f."""
    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)

        processer_list = []
        for _ in range(config.model.mp_num):
            processer_list.append(self.GN_Block(config, self.hidden_size))
        self.processer_list = torch.nn.ModuleList(processer_list)

        self.decoder = self.Decoder(config, self.hidden_size, self.output_sizes)
        self.integrator = self.Integrator(config, rho=1.0)

    @classmethod
    def get_feature_sizes(cls, dataset):
        return ([2, 5 + len(dataset.class_types), 0], [0, 5, 1])

    @classmethod
    def get_normalisation_map(cls):
        registry, inputs, outputs = super().get_normalisation_map()

        outputs.update({
            "cell_flux": (lambda outputs: outputs[0][:, 2:5], 'face_flux')
        })

        return registry, inputs, outputs

    def forward(self, graphs, mode='rollout'):
        graphs = self.normalizer.input(graphs)
        c_graph, f_graph, v_graph = graphs
        c_graph.edge_attr = f_graph.x
        c_graph_geom = c_graph.clone()

        c_graph = self.encoder(c_graph)

        for gnblock in self.processer_list:
            c_graph, v_graph = gnblock(c_graph, v_graph)

        edge_attr_out, vertex_out = self.decoder(c_graph, v_graph)

        cell_flux = calc_cell_flux_from_vertices(vertex_out, graphs)

        self.dt = c_graph_geom.dt
        acc_pred = self.integrator([cell_flux, edge_attr_out], c_graph_geom, f_graph, self.dt)
        output = [torch.cat([acc_pred, cell_flux], dim=1), edge_attr_out, None]

        if mode == "rollout":
            output = self.normalizer.output(output, inverse=True)

        return {
            'cell_velocity_change': output[0][:, 0:2],
            'cell_flux': output[0][:, 2:5],
            'face_velocity': output[1][:, 0:2],
            'face_pressure': output[1][:, 2:3],
        }

    class Integrator(torch.nn.Module):
        def __init__(self, config, rho):
            super().__init__()
            self.rho = rho
            self.face_area_norm = torch.nn.BatchNorm1d(1)  # or appropriate size
            self.vol_dt_norm = torch.nn.BatchNorm1d(1)
            self.face_area = None

        def forward(self, output, c_graph, f_graph, dt):
            unv = c_graph.normal
            rhs_coef = 1.0
            face_area = f_graph.area
            cell_face = f_graph.face

            edge_output = output[1]
            uv_face = edge_output[:, 0:2]
            p_face = edge_output[:, 2:3]
            flux_D = edge_output[:, 3:5]

            cell_flux = output[0]

            norm_coeff = normalize_vol_dt(c_graph.volume, c_graph.edge_index, dt, self.vol_dt_norm)
            n0 = norm_coeff[cell_face[0]]
            n1 = norm_coeff[cell_face[1]]
            n2 = norm_coeff[cell_face[2]]

            A0 = uv_face[cell_face[0]] * cell_flux[:, 0:1] * n0
            A1 = uv_face[cell_face[1]] * cell_flux[:, 1:2] * n1
            A2 = uv_face[cell_face[2]] * cell_flux[:, 2:3] * n2
            Phi_A = A0 + A1 + A2

            D0 = flux_D[cell_face[0], :]
            D1 = flux_D[cell_face[1], :]
            D2 = flux_D[cell_face[2], :]
            Phi_D = D0 + D1 + D2

            face_area = normalize_face_area(face_area, c_graph.volume, c_graph.edge_index, dt, self.face_area_norm)
            self.face_area = face_area # store for use elswhere in this step
            e0 = face_area[cell_face[0]]
            e1 = face_area[cell_face[1]]
            e2 = face_area[cell_face[2]]

            P0 = p_face[cell_face[0]] * unv[:, 0, :] * e0
            P1 = p_face[cell_face[1]] * unv[:, 1, :] * e1
            P2 = p_face[cell_face[2]] * unv[:, 2, :] * e2
            Phi_P = P0 + P1 + P2

            return rhs_coef * (-Phi_A - Phi_P / self.rho) + Phi_D

    def loss(self, output, graphs):
        c_graph, f_graph, v_graph = graphs

        loss_func = self.loss_func

        cell_divergence = fvm.divergence_from_cell_flux(output['cell_flux'])
        continuity   = loss_func(cell_divergence,
                                        torch.zeros_like(cell_divergence),
                                        None,
                                        c_graph.batch)
        cell_velocity_change     = loss_func(output['cell_velocity_change'],
                                        c_graph.y,
                                        None,
                                        c_graph.batch)
        face_velocity = loss_func(output['face_velocity'],
                                        f_graph.y[:, 0:2],
                                        None,
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

    class GN_Block(FvgnA.GN_Block):
        def __init__(self, config, hidden_size):
            super().__init__(config, hidden_size)
            # initializing message passing block objects
            self.edge_block = self.Face_Block(config, hidden_size)
            self.node_block = self.Cell_Block(config, hidden_size)
            self.vertex_block = self.Vertex_Block(config, hidden_size)

        def forward(self, c_graph, v_graph):
            # copying data to use in residual connection
            prev_edge_attr = c_graph.edge_attr.clone()
            prev_node_attr = c_graph.x.clone()

            # perform twice message passing
            c_graph = self.node_block(c_graph, v_graph)
            c_graph = self.edge_block(c_graph)
            v_graph = self.vertex_block(c_graph, v_graph)

            # residual connection
            edge_attr = prev_edge_attr + c_graph.edge_attr
            node_attr = prev_node_attr + c_graph.x
            vertex_attr = v_graph.x

            return Data(x=node_attr, edge_attr=edge_attr, edge_index=c_graph.edge_index), Data(x=vertex_attr, edge_index=v_graph.edge_index, face=v_graph.face)

        class Vertex_Block(torch.nn.Module):
            """Add all edge attr to connected vertices"""
            def __init__(self, config, hidden_size):
                super().__init__()

            def forward(self, cell_graph, vertex_graph):
                row, col = vertex_graph.edge_index
                vertex_indices = torch.cat([row, col], dim=0)  # shape: [2 * num_edges]
                edge_attrs = cell_graph.edge_attr.repeat(2, 1)  # shape: [2 * num_edges, attr_dim]
                vertex_edge_sum = scatter_add(edge_attrs, vertex_indices, dim=0, dim_size=cell_graph.x.size(0))
                return Data(x=vertex_edge_sum, edge_index=vertex_graph.edge_index, face=vertex_graph.face)

    class Decoder(torch.nn.Module):
        def __init__(self, config, hidden_size, output_sizes):
            super().__init__()
            self.edge_mlp = build_mlp(config, hidden_size, hidden_size, output_sizes[1], norm_layer=False)
            self.vertex_mlp = build_mlp(config, hidden_size, hidden_size, output_sizes[2], norm_layer=False)

        def forward(self, cell_graph, vertex_graph):
            return self.edge_mlp(cell_graph.edge_attr), self.vertex_mlp(vertex_graph.x)


class VertPotB(VertPotA):
    """VertPot A with physical integration"""
    face_grad_weights_use = True # needed for integrator

    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)

        self.integrator = self.Integrator(config, rho=1, nu=1e-3)
        self.face_mls_weights = MovingLeastSquaresWeights(config, loc='face')

    @classmethod
    def get_feature_sizes(cls, dataset):
        return ([2, 5 + len(dataset.class_types), 0], [0, 3, 1])

    def forward(self, graphs, mode='rollout'):
        graphs = self.normalizer.input(graphs)
        c_graph, f_graph, v_graph = graphs
        c_graph.edge_attr = f_graph.x
        c_graph_geom = c_graph.clone()

        c_graph = self.encoder(c_graph)

        for gnblock in self.processer_list:
            c_graph, v_graph = gnblock(c_graph, v_graph)

        edge_attr_out, vertex_out = self.decoder(c_graph, v_graph)
        cell_flux = calc_cell_flux_from_vertices(vertex_out, graphs)

        norm_cell_out = torch.cat([torch.zeros_like(c_graph_geom.x[:, 0:2]), cell_flux], dim=1)
        output = self.normalizer.output([norm_cell_out.clone(), edge_attr_out.clone(), None], inverse=True) # denormalise for integrator

        self.dt = c_graph_geom.dt
        acc_pred = self.integrator(output, c_graph_geom, f_graph, self.dt) # physical output
        output[0] = torch.cat([acc_pred, output[0][:, 2:]], dim=1)

        if mode != 'rollout': # normalised for training
            output = self.normalizer.output([torch.cat([acc_pred, torch.zeros_like(cell_flux)], dim=1), None, None]) # normalise for loss
            output[1] = edge_attr_out
            output[0][:, 2:5] = cell_flux
        else: # de norm
            output[0][:, 0:2]  = acc_pred

        return {
            'cell_velocity_change': output[0][:, 0:2],
            'cell_flux': output[0][:, 2:5],
            'face_velocity': output[1][:, 0:2],
            'face_pressure': output[1][:, 2:3],
        }

    class Integrator(torch.nn.Module):
        def __init__(self, config, rho, nu=1e-3):
            super().__init__()
            self.rho = rho
            self.nu = nu

        def forward(self, output, c_graph, f_graph, dt):
            unv = c_graph.normal
            face_area = f_graph.area
            cell_face = f_graph.face

            e0 = face_area[cell_face[0]]
            e1 = face_area[cell_face[1]]
            e2 = face_area[cell_face[2]]

            edge_output = output[1]
            uv_face = edge_output[:, 0:2]
            p_face = edge_output[:, 2:3]

            cell_flux = output[0][:, 2:5]
            A0 = uv_face[cell_face[0]] * cell_flux[:, 0:1]
            A1 = uv_face[cell_face[1]] * cell_flux[:, 1:2]
            A2 = uv_face[cell_face[2]] * cell_flux[:, 2:3]
            Phi_A = A0 + A1 + A2

            velocity_gradient = geometry.calc_gradient_tensor(uv_face, f_graph.grad_weights, f_graph.grad_neighbours)
            A0 = chain_flux_dot_product(velocity_gradient[cell_face[0]], unv[:, 0, :]) * e0
            A1 = chain_flux_dot_product(velocity_gradient[cell_face[1]], unv[:, 1, :]) * e1
            A2 = chain_flux_dot_product(velocity_gradient[cell_face[2]], unv[:, 2, :]) * e2
            Phi_D = A0 + A1 + A2

            P0 = p_face[cell_face[0]] * unv[:, 0, :] * e0
            P1 = p_face[cell_face[1]] * unv[:, 1, :] * e1
            P2 = p_face[cell_face[2]] * unv[:, 2, :] * e2
            Phi_P = P0 + P1 + P2

            return torch.mean(dt) / c_graph.volume * (-Phi_A - Phi_P / self.rho + self.nu * Phi_D)


class VertPotC(FluxC):
    """Predicts p_f and D_f on face and phi_f from difference of potentials at vertices. u_f is explicit."""
    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)

        processer_list = []
        for _ in range(config.model.mp_num):
            processer_list.append(VertPotA.GN_Block(config, self.hidden_size))
        self.processer_list = torch.nn.ModuleList(processer_list)

        self.decoder = VertPotA.Decoder(config, self.hidden_size, self.output_sizes)

        self.integrator = self.Integrator(config, rho=1.0)

    @classmethod
    def get_feature_sizes(cls, dataset):
        return ([2, 5 + len(dataset.class_types), 0], [0, 3, 1])

    def forward(self, graphs, mode='rollout'):
        graphs = self.normalizer.input(graphs)
        c_graph, f_graph, v_graph = graphs
        c_graph.edge_attr = f_graph.x
        c_graph_geom = c_graph.clone()

        c_graph = self.encoder(c_graph)

        for gnblock in self.processer_list:
            c_graph, v_graph = gnblock(c_graph, v_graph)

        edge_attr_out, vertex_out = self.decoder(c_graph, v_graph)

        cell_flux = calc_cell_flux_from_vertices(vertex_out, graphs)

        self.dt = c_graph_geom.dt
        acc_pred = self.integrator([cell_flux, edge_attr_out], c_graph_geom, f_graph, self.dt)
        output = [torch.cat([acc_pred, cell_flux], dim=1), edge_attr_out, None]

        if mode == "rollout":
            output = self.normalizer.output(output, inverse=True)

        return {
            'cell_velocity_change': output[0][:, 0:2],
            'cell_flux': output[0][:, 2:5],
            'face_pressure': output[1][:, 0:1],
        }

    class Integrator(torch.nn.Module):
        def __init__(self, config, rho):
            super().__init__()
            self.rho = rho
            self.face_area_norm = torch.nn.BatchNorm1d(1)
            self.face_area = None

        def forward(self, output, c_graph, f_graph, dt):
            unv = c_graph.normal
            rhs_coef = 1.0
            face_area = f_graph.area
            cell_face = f_graph.face

            edge_output = output[1]
            uv_face = geometry.cell_to_face_torch(c_graph.x[:, 0:2], c_graph.edge_index, f_graph.pos, c_graph.pos)
            p_face = edge_output[:, 0:1]
            flux_D = edge_output[:, 1:3]

            cell_flux = output[0]

            A0 = uv_face[cell_face[0]] * cell_flux[:, 0:1]
            A1 = uv_face[cell_face[1]] * cell_flux[:, 1:2]
            A2 = uv_face[cell_face[2]] * cell_flux[:, 2:3]
            Phi_A = A0 + A1 + A2

            D0 = flux_D[cell_face[0], :]
            D1 = flux_D[cell_face[1], :]
            D2 = flux_D[cell_face[2], :]
            Phi_D = D0 + D1 + D2

            face_area = normalize_face_area(face_area, c_graph.volume, c_graph.edge_index, dt, self.face_area_norm)
            self.face_area = face_area # store for use elsewhere in this step
            e0 = face_area[cell_face[0]]
            e1 = face_area[cell_face[1]]
            e2 = face_area[cell_face[2]]

            P0 = p_face[cell_face[0]] * unv[:, 0, :] * e0
            P1 = p_face[cell_face[1]] * unv[:, 1, :] * e1
            P2 = p_face[cell_face[2]] * unv[:, 2, :] * e2
            Phi_P = P0 + P1 + P2

            return rhs_coef * (-Phi_A - Phi_P / self.rho) + Phi_D

    def loss(self, output, graphs):
        c_graph, f_graph, v_graph = graphs

        loss_func = self.loss_func

        cell_divergence = fvm.divergence_from_cell_flux(output['cell_flux'])
        continuity   = loss_func(cell_divergence,
                                        torch.zeros_like(cell_divergence),
                                        None,
                                        c_graph.batch)
        cell_velocity_change     = loss_func(output['cell_velocity_change'],
                                        c_graph.y,
                                        None,
                                        c_graph.batch)
        # face_flux = loss_func(output['face_flux'],
        #                                 f_graph.y[:, 1:2],
        #                                 None,
        #                                 f_graph.batch)
        face_pressure = loss_func(output['face_pressure'],
                                        f_graph.y[:, 0:1],
                                        None,
                                        f_graph.batch)

        w = self.config.training.loss_weights
        total = w['continuity'] * continuity + w['cell_velocity_change'] * cell_velocity_change + w['face_pressure'] * face_pressure # + w['face_flux'] * face_flux
        loss = torch.mean(torch.log(total))

        return {
            "total_log_loss": loss,
            "continuity_loss": continuity,
            "cell_velocity_change_loss": cell_velocity_change,
            # "face_flux_loss": face_flux,
            "face_pressure_loss": face_pressure
        }


class VertPotD(FluxA):
    """Predicts u_f, p_f and flux_f. There IS loss on flux_f unlike VertPotA."""
    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)

        processer_list = []
        for _ in range(config.model.mp_num):
            processer_list.append(VertPotA.GN_Block(config, self.hidden_size))
        self.processer_list = torch.nn.ModuleList(processer_list)

        self.decoder = VertPotA.Decoder(config, self.hidden_size, self.output_sizes)

    @classmethod
    def get_feature_sizes(cls, dataset):
        return ([2, 5 + len(dataset.class_types), 0], [0, 5, 1])

    def forward(self, graphs, mode='rollout'):
        graphs = self.normalizer.input(graphs)
        c_graph, f_graph, v_graph = graphs
        c_graph.edge_attr = f_graph.x
        c_graph_geom = c_graph.clone()

        c_graph = self.encoder(c_graph)

        for gnblock in self.processer_list:
            c_graph, v_graph = gnblock(c_graph, v_graph)

        edge_attr_out, vertex_out = self.decoder(c_graph, v_graph)

        cell_flux = calc_cell_flux_from_vertices(vertex_out, graphs)
        face_flux = fvm.convert_cell_flux_to_face_flux_alt(cell_flux, c_graph.edge_index, f_graph.face)
        edge_attr_out = torch.cat([edge_attr_out[:, 0:3], face_flux, edge_attr_out[:, 3:5]], dim=1)

        self.dt = c_graph_geom.dt
        acc_pred = self.integrator(edge_attr_out, c_graph_geom, f_graph, self.dt)
        output = [acc_pred, edge_attr_out, None]

        if mode == 'rollout':
            output = self.normalizer.output(output, inverse=True)

        return {
            'cell_velocity_change': output[0][:, 0:2],
            'face_velocity': output[1][:, :2],
            'face_pressure': output[1][:, 2:3],
            'face_flux': output[1][:, 3:4],
        }

class VertPotE(FluxC):
    """Flux C with GN_Block from VertPotA."""
    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)

        processer_list = []
        for _ in range(config.model.mp_num):
            processer_list.append(VertPotA.GN_Block(config, self.hidden_size))
        self.processer_list = torch.nn.ModuleList(processer_list)

        self.decoder = VertPotA.Decoder(config, self.hidden_size, self.output_sizes)

    @classmethod
    def get_feature_sizes(cls, dataset):
        return ([2, 5 + len(dataset.class_types), 0], [0, 3, 1])

    def forward(self, graphs, mode='rollout'):
        graphs = self.normalizer.input(graphs)
        c_graph, f_graph, v_graph = graphs
        c_graph.edge_attr = f_graph.x
        c_graph_geom = c_graph.clone()

        c_graph = self.encoder(c_graph)

        for gnblock in self.processer_list:
            c_graph, v_graph = gnblock(c_graph, v_graph)

        edge_attr_out, vertex_out = self.decoder(c_graph, v_graph)

        cell_flux = calc_cell_flux_from_vertices(vertex_out, graphs)
        face_flux = fvm.convert_cell_flux_to_face_flux(cell_flux, c_graph.edge_index, f_graph.face)
        edge_attr_out = torch.cat([edge_attr_out, face_flux], dim=1)

        self.dt = c_graph_geom.dt
        acc_pred = self.integrator(edge_attr_out, c_graph_geom, f_graph, self.dt)
        output = [acc_pred, edge_attr_out, None]

        if mode == 'rollout':
            output = self.normalizer.output(output, inverse=True)

        return {
            'cell_velocity_change': output[0][:, 0:2],
            'face_velocity': output[1][:, :2],
            'face_pressure': output[1][:, 2:3],
            'face_flux': output[1][:, 3:4],
        }

class VertPotF(FluxA):
    """Physical integration"""
    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)

        processer_list = []
        for _ in range(config.model.mp_num):
            processer_list.append(VertPotA.GN_Block(config, self.hidden_size))
        self.processer_list = torch.nn.ModuleList(processer_list)

        self.decoder = VertPotA.Decoder(config, self.hidden_size, self.output_sizes)
        self.integrator = self.Integrator(config, rho=1.0)

    @classmethod
    def get_feature_sizes(cls, dataset):
        return ([2, 5 + len(dataset.class_types), 0], [0, 3, 1])

    def forward(self, graphs, mode='rollout'):
        graphs = self.normalizer.input(graphs)
        c_graph, f_graph, v_graph = graphs
        c_graph.edge_attr = f_graph.x
        c_graph_geom = c_graph.clone()

        c_graph = self.encoder(c_graph)

        for gnblock in self.processer_list:
            c_graph, v_graph = gnblock(c_graph, v_graph)

        edge_attr_out, vertex_out = self.decoder(c_graph, v_graph)

        cell_flux = calc_cell_flux_from_vertices(vertex_out, graphs)
        face_flux = fvm.convert_cell_flux_to_face_flux_alt(cell_flux, c_graph.edge_index, f_graph.face)
        edge_attr_out = torch.cat([edge_attr_out, face_flux], dim=1)

        output = self.normalizer.output([None, edge_attr_out.clone(), None], inverse=True) # denormalise for integrator
        self.dt = c_graph_geom.dt
        acc_pred = self.integrator(edge_attr_out, c_graph_geom, f_graph, self.dt)
        output = [acc_pred, edge_attr_out, None]

        if mode != 'rollout': # normalised for training
            output = self.normalizer.output([acc_pred, None, None]) # normalise for loss
            output[1] = edge_attr_out
        else: # de norm
            output[0]  = acc_pred

        return {
            'cell_velocity_change': output[0][:, 0:2],
            'face_velocity': output[1][:, :2],
            'face_pressure': output[1][:, 2:3],
            'face_flux': output[1][:, 3:4],
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

            uv_face = edge_output[:, :2]
            p_face = edge_output[:, 2:3]
            flux_face = edge_output[:, 3:4]

            A0 = uv_face[cell_face[0]] * flux_face[cell_face[0]]
            A1 = uv_face[cell_face[1]] * flux_face[cell_face[1]]
            A2 = uv_face[cell_face[2]] * flux_face[cell_face[2]]
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


class VertPotG(FluxA):
    """Loss on flux"""
    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)

        processer_list = []
        for _ in range(config.model.mp_num):
            processer_list.append(self.GN_Block(config, self.hidden_size))
        self.processer_list = torch.nn.ModuleList(processer_list)

        self.decoder = self.Decoder(config, self.hidden_size, self.output_sizes)
        self.integrator = self.Integrator(config, rho=1.0)

    @classmethod
    def get_feature_sizes(cls, dataset):
        return ([2, 5 + len(dataset.class_types), 0], [0, 5, 1])

    @classmethod
    def get_normalisation_map(cls):
        registry, inputs, outputs = super().get_normalisation_map()

        outputs.update({
            "cell_flux": (lambda outputs: outputs[0][:, 2:5], 'face_flux')
        })

        return registry, inputs, outputs

    def forward(self, graphs, mode='rollout'):
        graphs = self.normalizer.input(graphs)
        c_graph, f_graph, v_graph = graphs
        c_graph.edge_attr = f_graph.x
        c_graph_geom = c_graph.clone()

        c_graph = self.encoder(c_graph)

        for gnblock in self.processer_list:
            c_graph, v_graph = gnblock(c_graph, v_graph)

        edge_attr_out, vertex_out = self.decoder(c_graph, v_graph)

        cell_flux = calc_cell_flux_from_vertices(vertex_out, graphs)

        self.dt = c_graph_geom.dt
        acc_pred = self.integrator([cell_flux, edge_attr_out], c_graph_geom, f_graph, self.dt)
        output = [torch.cat([acc_pred, cell_flux], dim=1), edge_attr_out, None]

        if mode == "rollout":
            output = self.normalizer.output(output, inverse=True)

        face_flux = geometry.cell_flux_to_face_flux(output[0][:, 2:5], c_graph.edge_index, f_graph.face)

        return {
            'cell_velocity_change': output[0][:, 0:2],
            'face_flux': face_flux,
            'face_velocity': output[1][:, 0:2],
            'face_pressure': output[1][:, 2:3],
        }

    class Integrator(torch.nn.Module):
        def __init__(self, config, rho):
            super().__init__()
            self.rho = rho
            self.face_area_norm = torch.nn.BatchNorm1d(1)  # or appropriate size
            self.vol_dt_norm = torch.nn.BatchNorm1d(1)
            self.face_area = None

        def forward(self, output, c_graph, f_graph, dt):
            unv = c_graph.normal
            rhs_coef = 1.0
            face_area = f_graph.area
            cell_face = f_graph.face

            edge_output = output[1]
            uv_face = edge_output[:, 0:2]
            p_face = edge_output[:, 2:3]
            flux_D = edge_output[:, 3:5]

            cell_flux = output[0]

            norm_coeff = normalize_vol_dt(c_graph.volume, c_graph.edge_index, dt, self.vol_dt_norm)
            n0 = norm_coeff[cell_face[0]]
            n1 = norm_coeff[cell_face[1]]
            n2 = norm_coeff[cell_face[2]]

            A0 = uv_face[cell_face[0]] * cell_flux[:, 0:1] * n0
            A1 = uv_face[cell_face[1]] * cell_flux[:, 1:2] * n1
            A2 = uv_face[cell_face[2]] * cell_flux[:, 2:3] * n2
            Phi_A = A0 + A1 + A2

            D0 = flux_D[cell_face[0], :]
            D1 = flux_D[cell_face[1], :]
            D2 = flux_D[cell_face[2], :]
            Phi_D = D0 + D1 + D2

            face_area = normalize_face_area(face_area, c_graph.volume, c_graph.edge_index, dt, self.face_area_norm)
            self.face_area = face_area # store for use elswhere in this step
            e0 = face_area[cell_face[0]]
            e1 = face_area[cell_face[1]]
            e2 = face_area[cell_face[2]]

            P0 = p_face[cell_face[0]] * unv[:, 0, :] * e0
            P1 = p_face[cell_face[1]] * unv[:, 1, :] * e1
            P2 = p_face[cell_face[2]] * unv[:, 2, :] * e2
            Phi_P = P0 + P1 + P2

            return rhs_coef * (-Phi_A - Phi_P / self.rho) + Phi_D

    def loss(self, output, graphs):
        c_graph, f_graph, v_graph = graphs

        loss_func = self.loss_func

        cell_divergence = fvm.divergence_from_face_flux(output['face_flux'], f_graph.face)
        continuity   = loss_func(cell_divergence,
                                        torch.zeros_like(cell_divergence),
                                        None,
                                        c_graph.batch)
        cell_velocity_change     = loss_func(output['cell_velocity_change'],
                                        c_graph.y,
                                        None,
                                        c_graph.batch)
        face_velocity = loss_func(output['face_velocity'],
                                        f_graph.y[:, 0:2],
                                        None,
                                        f_graph.batch)
        face_pressure = loss_func(output['face_pressure'],
                                        f_graph.y[:, 2:3],
                                        None,
                                        f_graph.batch)
        face_flux_loss = loss_func(output['face_flux'],f_graph.y[:, 3:4], None, f_graph.batch)

        w = self.config.training.loss_weights
        total = w['continuity'] * continuity + w['cell_velocity_change'] * cell_velocity_change + w['face_velocity'] * face_velocity + w['face_pressure'] * face_pressure + w['face_flux'] * face_flux_loss
        loss = torch.mean(torch.log(total))

        return {
            "total_log_loss": loss,
            "continuity_loss": continuity,
            "cell_velocity_change_loss": cell_velocity_change,
            "face_velocity_loss": face_velocity,
            "face_pressure_loss": face_pressure
        }

    class GN_Block(FvgnA.GN_Block):
        def __init__(self, config, hidden_size):
            super().__init__(config, hidden_size)
            # initializing message passing block objects
            self.edge_block = self.Face_Block(config, hidden_size)
            self.node_block = self.Cell_Block(config, hidden_size)
            self.vertex_block = self.Vertex_Block(config, hidden_size)

        def forward(self, c_graph, v_graph):
            # copying data to use in residual connection
            prev_edge_attr = c_graph.edge_attr.clone()
            prev_node_attr = c_graph.x.clone()

            # perform twice message passing
            c_graph = self.node_block(c_graph, v_graph)
            c_graph = self.edge_block(c_graph)
            v_graph = self.vertex_block(c_graph, v_graph)

            # residual connection
            edge_attr = prev_edge_attr + c_graph.edge_attr
            node_attr = prev_node_attr + c_graph.x
            vertex_attr = v_graph.x

            return Data(x=node_attr, edge_attr=edge_attr, edge_index=c_graph.edge_index), Data(x=vertex_attr, edge_index=v_graph.edge_index, face=v_graph.face)

        class Vertex_Block(torch.nn.Module):
            """Add all edge attr to connected vertices"""
            def __init__(self, config, hidden_size):
                super().__init__()

            def forward(self, cell_graph, vertex_graph):
                row, col = vertex_graph.edge_index
                vertex_indices = torch.cat([row, col], dim=0)  # shape: [2 * num_edges]
                edge_attrs = cell_graph.edge_attr.repeat(2, 1)  # shape: [2 * num_edges, attr_dim]
                vertex_edge_sum = scatter_add(edge_attrs, vertex_indices, dim=0, dim_size=cell_graph.x.size(0))
                return Data(x=vertex_edge_sum, edge_index=vertex_graph.edge_index, face=vertex_graph.face)

    class Decoder(torch.nn.Module):
        def __init__(self, config, hidden_size, output_sizes):
            super().__init__()
            self.edge_mlp = build_mlp(config, hidden_size, hidden_size, output_sizes[1], norm_layer=False)
            self.vertex_mlp = build_mlp(config, hidden_size, hidden_size, output_sizes[2], norm_layer=False)

        def forward(self, cell_graph, vertex_graph):
            return self.edge_mlp(cell_graph.edge_attr), self.vertex_mlp(vertex_graph.x)
