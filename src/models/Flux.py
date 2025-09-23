# === LIBRARIES ===


import re
import torch


# === MODULES ===


from models.Fvgn import FvgnA
from utils.normalisation import normalize_face_area, normalize_vol_dt
from utils.maths import chain_dot_product, chain_flux_dot_product
import utils.geometry as geometry
import utils.transforms as transforms
import utils.fvm as fvm


# === FUNCTIONS ===





# === CLASSES ===


class FluxA(FvgnA):
    """
    Flux-based FVGN variant that predicts face velocities and fluxes.
    Extends FVGN to jointly predict velocity and flux fields with combined loss.
    """
    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)
        self.integrator = self.Integrator(config, rho=1.0)

    @classmethod
    def get_feature_sizes(cls, dataset):
        return ([2, 5 + len(dataset.class_types), 0], [0, 6, 0])

    @classmethod
    def get_normalisation_map(cls):
        registry, inputs, outputs = super().get_normalisation_map()

        registry.update({
            "face_flux": (lambda graphs: graphs[1].y[:, 3:4], 'z_score')
        })

        inputs.update({
            "face_flux": (lambda graphs: graphs[1].y[:, 3:4], 'face_flux')
        })

        outputs.update({
            "face_flux": (lambda outputs: outputs[1][:, 3:4], 'face_flux')
        })

        return registry, inputs, outputs

    @classmethod
    def transform_features(cls, dataset, graphs, mesh_id=None):
        cell_graph, face_graph, vertex_graph = graphs
        cell_velocity = cell_graph.velocity[:, 0]
        cell_velocity_target = cell_graph.velocity[:, -1]
        if dataset.noise:
            cell_velocity = transforms.add_noise(cell_velocity, std=dataset.config.training.noise_std)
        cell_graph.x = cell_velocity
        cell_graph.y = cell_velocity_target - cell_velocity

        # Flip edges
        if dataset.mode == 'train':
            cell_graph.edge_index, flip_mask = transforms.random_edge_flip(cell_graph.edge_index)
            safe_flip = flip_mask & (cell_graph.edge_index[0] != cell_graph.edge_index[1]) # don't flip on boundaries
            face_graph.normal[safe_flip] *= -1
            face_graph.flux[safe_flip] *= -1 # flip all fluxes

        # Use FVGN-like boundaries
        face_type = face_graph.type
        f_interior_mask = (face_type == dataset.class_types.NORMAL) | (face_type == dataset.class_types.OUTFLOW) | (face_type == dataset.class_types.SLIP) | (face_type == dataset.class_types.WALL_BOUNDARY)
        face_graph.boundary_mask = ~f_interior_mask.squeeze()

        face_velocity_change = transforms.calc_face_velocity_change(cell_velocity, cell_graph.edge_index)
        face_edge_vector  = transforms.calc_cell_edge_vector(cell_graph.pos, cell_graph.edge_index)
        face_type_one_hot = transforms.calc_face_type_one_hot(face_graph.type, len(dataset.class_types))
        face_graph.x = torch.cat([face_velocity_change, face_edge_vector, face_graph.area, face_type_one_hot], dim=1)
        face_graph.y = torch.cat([face_graph.velocity[:, -1], face_graph.pressure[:, -1], face_graph.flux[:, -1]], dim=1)

        return (cell_graph, face_graph, vertex_graph)

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

        cell_flux = fvm.face_flux_to_cell_flux_vectorized(output[1][:, 3:4], f_graph.face, c_graph.edge_index)
        return {
            'cell_velocity_change': output[0][:, 0:2],
            'face_velocity': output[1][:, 0:2],
            'face_pressure': output[1][:, 2:3],
            'face_flux': output[1][:, 3:4],
            'cell_flux': cell_flux.squeeze(-1)  # (num_cells, 3, 1)
        }

    def loss(self, output, graphs):
        c_graph, f_graph, v_graph = graphs

        loss_func = self.loss_func

        cell_divergence = fvm.divergence_from_cell_flux(output['cell_flux'])
        continuity   = loss_func(cell_divergence,
                                        torch.zeros_like(cell_divergence),
                                        None,
                                        c_graph.batch)
        cell_velocity_change = loss_func(output['cell_velocity_change'],
                                        c_graph.y,
                                        None,
                                        c_graph.batch)
        face_velocity = loss_func(output['face_velocity'],
                                        f_graph.y[:, :2],
                                        ~f_graph.boundary_mask, # only interior
                                        f_graph.batch)
        face_flux = loss_func(output['face_flux'],
                                        f_graph.y[:, 3:4],
                                        None,
                                        f_graph.batch)
        face_pressure = loss_func(output['face_pressure'],
                                        f_graph.y[:, 2:3],
                                        None,
                                        f_graph.batch)

        w = self.config.training.loss_weights
        total = w['continuity'] * continuity + w['cell_velocity_change'] * cell_velocity_change + w['face_velocity'] * face_velocity + w['face_flux'] * face_flux + w['face_pressure'] * face_pressure
        loss = torch.mean(torch.log(total))

        return {
            "total_log_loss": loss,
            "continuity_loss": continuity,
            "cell_velocity_change_loss": cell_velocity_change,
            "face_velocity_loss": face_velocity,
            "face_flux_loss": face_flux,
            "face_pressure_loss": face_pressure
        }

    class Integrator(torch.nn.Module):
        def __init__(self, config, rho):
            super().__init__()
            self.rho = rho
            self.face_area_norm = torch.nn.BatchNorm1d(1)
            self.vol_dt_norm = torch.nn.BatchNorm1d(1)
            self.face_area = None

        def forward(self, edge_output, c_graph, f_graph, dt):
            unv = c_graph.normal
            rhs_coef = 1.0
            face_area = f_graph.area
            cell_face = f_graph.face
            cell_edge_index = c_graph.edge_index

            uv_face = edge_output[:, :2]
            p_face = edge_output[:, 2:3]
            flux_face = edge_output[:, 3:4]
            flux_D = edge_output[:, 4:6]

            cell_flux = fvm.face_flux_to_cell_flux_vectorized(flux_face, cell_face, cell_edge_index)

            norm_coeff = normalize_vol_dt(c_graph.volume, c_graph.edge_index, dt, self.vol_dt_norm)
            n0 = norm_coeff[cell_face[0]]
            n1 = norm_coeff[cell_face[1]]
            n2 = norm_coeff[cell_face[2]]

            A0 = uv_face[cell_face[0]] * cell_flux[:, 0] * n0
            A1 = uv_face[cell_face[1]] * cell_flux[:, 1] * n1
            A2 = uv_face[cell_face[2]] * cell_flux[:, 2] * n2
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


class FluxB(FluxA):
    """Predicts u_f only and takes loss of flux_f"""
    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)
        self.integrator = FvgnA.Integrator(config, rho=1.0)

    @classmethod
    def get_feature_sizes(cls, dataset):
        return ([2, 5 + len(dataset.class_types), 0], [0, 5, 0])

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

        face_area = self.integrator.face_area
        if mode == 'rollout':
            output = self.normalizer.output(output, inverse=True)
            face_area = f_graph.area

        face_flux = fvm.calc_flux_from_uf(output[1][:, 0:2], f_graph.normal, face_area) #Q: denorm and then calc or calc and then denorm?

        return {
            'cell_velocity_change': output[0][:, 0:2],
            'face_velocity': output[1][:, 0:2],
            'face_pressure': output[1][:, 2:3],
            'face_flux': face_flux,
        }

    def loss(self, output, graphs):
        c_graph, f_graph, v_graph = graphs

        loss_func = self.loss_func

        cell_divergence = fvm.divergence_from_face_flux(output['face_flux'], f_graph.face)
        continuity   = loss_func(cell_divergence,
                                        torch.zeros_like(cell_divergence),
                                        None,
                                        c_graph.batch)
        cell_velocity_change = loss_func(output['cell_velocity_change'],
                                        c_graph.y[:, 0:2],
                                        None,
                                        c_graph.batch)
        face_flux = loss_func(output['face_flux'],
                                        f_graph.y[:, 3:4],
                                        None,
                                        f_graph.batch)
        face_pressure = loss_func(output['face_pressure'],
                                        f_graph.y[:, 2:3],
                                        None,
                                        f_graph.batch)

        w = self.config.training.loss_weights
        total = w['continuity'] * continuity + w['cell_velocity_change'] * cell_velocity_change + w['face_flux'] * face_flux + w['face_pressure'] * face_pressure
        loss = torch.mean(torch.log(total))

        return {
            "total_log_loss": loss,
            # "continuity_loss": continuity,
            "cell_velocity_change_loss": cell_velocity_change,
            "face_flux_loss": face_flux,
            "face_pressure_loss": face_pressure
        }


class FluxC(FvgnA):
    """Predicts only phi and uses explicit u_f"""
    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)
        self.integrator = self.Integrator(config, rho=1.0)

    @classmethod
    def get_feature_sizes(cls, dataset):
        return ([2, 5 + len(dataset.class_types), 0], [0, 4, 0])

    @classmethod
    def transform_features(cls, dataset, graphs, mesh_id=None):
        cell_graph, face_graph, vertex_graph = graphs
        cell_velocity = cell_graph.velocity[:, 0]
        cell_velocity_target = cell_graph.velocity[:, -1]
        if dataset.noise:
            cell_velocity = transforms.add_noise(cell_velocity, std=dataset.config.training.noise_std)
        cell_graph.x = cell_velocity
        cell_graph.y = cell_velocity_target - cell_velocity

        # Flip edges
        if dataset.mode == 'train':
            cell_graph.edge_index, flip_mask = transforms.random_edge_flip(cell_graph.edge_index)
            safe_flip = flip_mask & (cell_graph.edge_index[0] != cell_graph.edge_index[1]) # don't flip on boundaries
            face_graph.normal[safe_flip] *= -1
            face_graph.flux[:, -1][safe_flip] *= -1

        # Use FVGN-like boundaries
        face_type = face_graph.type
        f_interior_mask = (face_type == dataset.class_types.NORMAL) | (face_type == dataset.class_types.OUTFLOW) | (face_type == dataset.class_types.SLIP) | (face_type == dataset.class_types.WALL_BOUNDARY)
        face_graph.boundary_mask = ~f_interior_mask.squeeze()

        face_velocity_change = transforms.calc_face_velocity_change(cell_velocity, cell_graph.edge_index)
        face_edge_vector  = transforms.calc_cell_edge_vector(cell_graph.pos, cell_graph.edge_index)
        face_type_one_hot = transforms.calc_face_type_one_hot(face_graph.type, len(dataset.class_types))
        face_graph.x = torch.cat([face_velocity_change, face_edge_vector, face_graph.area, face_type_one_hot], dim=1)
        face_graph.y = torch.cat([face_graph.pressure[:, -1], face_graph.flux[:, -1]], dim=1)

        return (cell_graph, face_graph, vertex_graph)

    @classmethod
    def get_normalisation_map(cls):
        registry, inputs, outputs = super().get_normalisation_map()

        del registry['face_velocity_x']
        del registry['face_velocity_y']
        del registry['face_pressure']
        registry.update({
            "face_pressure": (lambda graphs: graphs[1].y[:, 0:1], 'z_score'),
            "face_flux": (lambda graphs: graphs[1].y[:, 1:2], 'z_score')
        })

        del inputs['face_velocity_x']
        del inputs['face_velocity_y']
        del inputs['face_pressure']
        inputs.update({
            "face_pressure": (lambda graphs: graphs[1].y[:, 0:1], 'face_pressure'),
            "face_flux": (lambda graphs: graphs[1].y[:, 1:2], 'face_flux')
        })

        del outputs['face_velocity_x']
        del outputs['face_velocity_y']
        del outputs['face_pressure']
        outputs.update({
            "face_pressure": (lambda outputs: outputs[1][:, 0:1], 'face_pressure'),
            "face_flux": (lambda outputs: outputs[1][:, 1:2], 'face_flux')
        })

        return registry, inputs, outputs

    def forward(self, graphs, mode="rollout"):
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
            'face_pressure': output[1][:, 0:1],
            'face_flux': output[1][:, 1:2],
        }

    class Integrator(torch.nn.Module):
        def __init__(self, config, rho):
            super().__init__()
            self.rho = rho
            self.face_area_norm = torch.nn.BatchNorm1d(1)  # or appropriate size
            self.face_area = None

        def forward(self, edge_output, c_graph, f_graph, dt):
            unv = c_graph.normal
            rhs_coef = 1.0
            face_area = f_graph.area
            cell_face = f_graph.face

            uv_face = geometry.cell_to_face_torch(c_graph.x[:, 0:2], c_graph.edge_index, f_graph.pos, c_graph.pos)
            p_face = edge_output[:, 0:1]
            flux_face = edge_output[:, 1:2]
            flux_D = edge_output[:, 2:4]

            A0 = uv_face[cell_face[0]] * flux_face[cell_face[0]]
            A1 = uv_face[cell_face[1]] * flux_face[cell_face[1]]
            A2 = uv_face[cell_face[2]] * flux_face[cell_face[2]]
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
        cell_velocity_change = loss_func(output['cell_velocity_change'],
                                        c_graph.y[:, 0:2],
                                        None,
                                        c_graph.batch)
        face_flux = loss_func(output['face_flux'],
                                        f_graph.y[:, 1:2],
                                        None,
                                        f_graph.batch)
        face_pressure = loss_func(output['face_pressure'],
                                        f_graph.y[:, 0:1],
                                        None,
                                        f_graph.batch)

        w = self.config.training.loss_weights
        total = w['continuity'] * continuity + w['cell_velocity_change'] * cell_velocity_change + w['face_flux'] * face_flux + w['face_pressure'] * face_pressure
        loss = torch.mean(torch.log(total))

        return {
            "total_log_loss": loss,
            "continuity_loss": continuity,
            "cell_velocity_change_loss": cell_velocity_change,
            "face_flux_loss": face_flux,
            "face_pressure_loss": face_pressure
        }


class FluxD(FluxA):
    """Physical integration with adpative denorm (FvgnJ-style)"""
    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)
        self.integrator = self.Integrator(config, rho=1.0)

        self.velocity_scale_x = torch.nn.Parameter(torch.tensor(0.1))
        self.velocity_scale_y = torch.nn.Parameter(torch.tensor(0.0001))
        self.pressure_scale = torch.nn.Parameter(torch.tensor(0.01))
        self.diffusion_scale = torch.nn.Parameter(torch.tensor(0.01))
        self.flux_scale = torch.nn.Parameter(torch.tensor(0.001))

        self.velocity_bias_x = 0#torch.nn.Parameter(torch.tensor(0.0))
        self.velocity_bias_y = 0#torch.nn.Parameter(torch.tensor(0.0))
        self.pressure_bias = 0#torch.nn.Parameter(torch.tensor(0.0))
        self.diffusion_bias = 0#torch.nn.Parameter(torch.tensor(0.0))
        self.flux_bias = 0#torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, graphs, mode='rollout'):
        # f_face = graphs[1].y[:, 3:4].clone()

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
        f_face = edge_attr_out_raw[:, 3:4] * self.flux_scale + self.flux_bias
        d_face = edge_attr_out_raw[:, 4:6] * self.diffusion_scale + self.diffusion_bias

        edge_attr_out = torch.cat([uv_face, p_face, f_face, d_face], dim=-1)
        output = [None, edge_attr_out, None]
        self.dt = c_graph_geom.dt
        acc_pred = self.integrator(output[1], c_graph_geom, f_graph, self.dt) # outputs denormalised

        output = [acc_pred, edge_attr_out, None]
        if mode != 'rollout': # normalised for training
            output = self.normalizer.output(output) # normalise for loss

        cell_flux = fvm.face_flux_to_cell_flux_vectorized(output[1][:, 3:4], f_graph.face, c_graph.edge_index)
        return {
            'cell_velocity_change': output[0][:, 0:2],
            'face_velocity': output[1][:, 0:2],
            'face_pressure': output[1][:, 2:3],
            'face_flux': output[1][:, 3:4],
            'cell_flux': cell_flux.squeeze(-1)  # (num_cells, 3)
        }

    def loss(self, output, graphs):
        c_graph, f_graph, v_graph = graphs

        loss_func = self.loss_func

        cell_divergence = fvm.divergence_from_cell_flux(output['cell_flux'])
        continuity   = loss_func(cell_divergence,
                                        torch.zeros_like(cell_divergence),
                                        None,
                                        c_graph.batch)
        cell_velocity_change = loss_func(output['cell_velocity_change'],
                                        c_graph.y,
                                        None,
                                        c_graph.batch)
        face_velocity = loss_func(output['face_velocity'],
                                        f_graph.y[:, :2],
                                        ~f_graph.boundary_mask, # only interior
                                        f_graph.batch)
        face_flux = loss_func(output['face_flux'],
                                        f_graph.y[:, 3:4],
                                        None,
                                        f_graph.batch)
        face_pressure = loss_func(output['face_pressure'],
                                        f_graph.y[:, 2:3],
                                        None,
                                        f_graph.batch)

        w = self.config.training.loss_weights
        total = w['continuity'] * continuity + w['cell_velocity_change'] * cell_velocity_change + w['face_velocity'] * face_velocity + w['face_flux'] * face_flux + w['face_pressure'] * face_pressure
        loss = torch.mean(torch.log(total))

        return {
            "total_log_loss": loss,
            "continuity_loss": continuity,
            "cell_velocity_change_loss": cell_velocity_change,
            "face_velocity_loss": face_velocity,
            "face_flux_loss": face_flux,
            "face_pressure_loss": face_pressure
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
            cell_edge_index = c_graph.edge_index

            uv_face = edge_output[:, :2]
            p_face = edge_output[:, 2:3]
            flux_face = edge_output[:, 3:4]
            flux_D = edge_output[:, 4:6]

            cell_flux = fvm.face_flux_to_cell_flux_vectorized(flux_face, cell_face, cell_edge_index)

            A0 = uv_face[cell_face[0]] * cell_flux[:, 0]
            A1 = uv_face[cell_face[1]] * cell_flux[:, 1]
            A2 = uv_face[cell_face[2]] * cell_flux[:, 2]
            Phi_A = A0 + A1 + A2

            D0 = flux_D[cell_face[0], :]
            D1 = flux_D[cell_face[1], :]
            D2 = flux_D[cell_face[2], :]
            Phi_D = D0 + D1 + D2

            e0 = face_area[cell_face[0]]
            e1 = face_area[cell_face[1]]
            e2 = face_area[cell_face[2]]

            P0 = p_face[cell_face[0]] * unv[:, 0, :] * e0
            P1 = p_face[cell_face[1]] * unv[:, 1, :] * e1
            P2 = p_face[cell_face[2]] * unv[:, 2, :] * e2
            Phi_P = P0 + P1 + P2

            return torch.mean(dt) / c_graph.volume * (-Phi_A - Phi_P / self.rho + self.nu * Phi_D )
