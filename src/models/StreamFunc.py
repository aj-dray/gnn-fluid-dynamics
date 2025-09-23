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
from models.Model import Model, build_mlp
from models.Mgn import MgnA, MgnC, MgnB
from datasets.OpenFoam import NodeType
import h5py
import numpy as np


# === FUNCTIONS ===



# === CLASSES ===


class BaseStreamFunc:
    """Base class for potential-based models with common functionality"""

    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)
        self.divergence_layer = self.DivergenceLayer()
        self.cell_grad_weights_use = True
        self.cell_mls_weights = MovingLeastSquaresWeights(config, loc='cell')

    @classmethod
    def get_feature_sizes(cls, dataset):
        return ([2, 5 + len(dataset.class_types), 0], [2, 0, 0])

    def loss(self, output, graphs):
        """Standard loss computation for potential-based models"""
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
        total_loss = w['cell_velocity'] * cell_velocity_loss + w['cell_pressure'] * cell_pressure_loss
        total_log_loss = torch.mean(torch.log(total_loss))

        return {
            "total_log_loss": total_log_loss,
            "cell_velocity_loss": cell_velocity_loss,
            "cell_pressure_loss": cell_pressure_loss,
            "continuity_loss": continuity
        }

    def update_features(self, output, input_graphs):
        """Update input graphs with predicted outputs for autoregressive rollout"""
        c_graph, f_graph, v_graph = input_graphs

        # Update cell velocity with predicted velocity
        c_graph.x = output["cell_velocity"].detach()

        face_velocity_change = transforms.calc_face_velocity_change(c_graph.x[:, :2], c_graph.edge_index)
        face_type = f_graph.type
        mask = (face_type == NodeType.INFLOW) | (face_type == NodeType.WALL_BOUNDARY)
        face_velocity_change[mask] = f_graph.y[:, 0:2][mask] # apply boundary conditions manually

        f_graph.x[:, 0:2] = face_velocity_change

        return [c_graph, f_graph, v_graph]

    class DivergenceLayer(torch.nn.Module):
        """Find the gradient in x and y directions using MLS"""
        def __init__(self):
            super().__init__()

        def forward(self, cell_potential, weights, neighbours):
            neighbour_values = cell_potential[neighbours]  # [n_cells, k_neighbors]
            potential_diff = neighbour_values - cell_potential[:, None] # [n_cells, k_neighbors]
            # potential_diff = neighbour_values

            gradient_x = torch.sum(weights[:, :, 0] * potential_diff, dim=1)
            gradient_y = torch.sum(weights[:, :, 1] * potential_diff, dim=1)
            rotated_gradient = torch.stack([-gradient_y, +gradient_x], dim=1)
            return rotated_gradient


class StreamFuncA(BaseStreamFunc, MgnC):
    """Applies normalization early and creates divergence-free velocity in normalized space"""

    def forward(self, graphs, mode='rollout'):
        graphs = self.normalizer.input(graphs)
        c_graph, f_graph, v_graph = graphs
        c_graph.edge_attr = f_graph.x
        c_graph_save = c_graph.clone()

        # Encoder-Processor-Decoder
        c_graph = self.encoder(c_graph)
        for gnblock in self.processer_list:
            c_graph = gnblock(c_graph, v_graph)
        cell_output = self.decoder(c_graph)

        # create divergence-free velocity field
        cell_velocity = self.divergence_layer(cell_output[:, 0], c_graph_save.grad_weights, c_graph_save.grad_neighbours)

        output = (torch.cat([cell_velocity, cell_output[:, 1:2]], dim=1), None, None)
        # Normalisation
        if mode == 'rollout':
            output = self.normalizer.output(output, inverse=True)

        return {
            'cell_velocity': output[0][:, 0:2],
            'cell_pressure': output[0][:, 2:3],
        }


class StreamFuncB(BaseStreamFunc, MgnC):
    """Creates divergence-free velocity in denormalized space, then renormalizes for training"""

    def forward(self, graphs, mode='rollout'):
        graphs = self.normalizer.input(graphs)
        c_graph, f_graph, v_graph = graphs
        c_graph.edge_attr = f_graph.x
        c_graph_save = c_graph.clone()

        # Encoder-Processor-Decoder
        c_graph = self.encoder(c_graph)
        for gnblock in self.processer_list:
            c_graph = gnblock(c_graph, v_graph)
        cell_output = self.decoder(c_graph)

        cell_output_expanded = torch.cat([cell_output[:, 0:1], torch.zeros_like(cell_output[:, 0:1]), cell_output[:, 1:2]], dim=1)
        output_norm = (cell_output_expanded, None, None)
        output = self.normalizer.output(output_norm, inverse=True)

        # create divergence-free velocity field
        cell_velocity = self.divergence_layer(output[0][:, 0], c_graph_save.grad_weights, c_graph_save.grad_neighbours)
        output[0][:, 0:2] = cell_velocity

        if mode == 'train': # for loss
            output = self.normalizer.output(output, inverse=False)

        return {
            'cell_velocity': output[0][:, 0:2],
            'cell_pressure': output[0][:, 2:3],
        }


class StreamFuncC(BaseStreamFunc, MgnB):
    """Simplified approach without normalization in forward method - inherits from MgnB instead of MgnC"""

    def forward(self, graphs, mode='rollout'):
        c_graph, f_graph, v_graph = graphs
        c_graph.edge_attr = f_graph.x
        c_graph_save = c_graph.clone()

        # Encoder-Processor-Decoder
        c_graph = self.encoder(c_graph)
        for gnblock in self.processer_list:
            c_graph = gnblock(c_graph, v_graph)
        cell_output = self.decoder(c_graph)

        # create divergence-free velocity field
        cell_velocity = self.divergence_layer(cell_output[:, 0], c_graph_save.grad_weights, c_graph_save.grad_neighbours)

        output = (torch.cat([cell_velocity, cell_output[:, 1:2]], dim=1), None, None)

        return {
            'cell_velocity': output[0][:, 0:2],
            'cell_pressure': output[0][:, 2:3],
        }


class StreamFuncD(BaseStreamFunc, MgnC):
    """Adds potential smoothing and includes potential in output for additional regularization"""

    cell_grad_weights_use = True

    def __init__(self, config, loss_func, dataset, stats):
        super().__init__(config, loss_func, dataset, stats)
        neighbours = self.cell_mls_weights
        self.smoother = self.SmoothingLayer(neighbours=8)

    def forward(self, graphs, mode='rollout'):
        graphs = self.normalizer.input(graphs)
        c_graph, f_graph, v_graph = graphs
        c_graph.edge_attr = f_graph.x
        c_graph_save = c_graph.clone()

        # Encoder-Processor-Decoder
        c_graph = self.encoder(c_graph)
        for gnblock in self.processer_list:
            c_graph = gnblock(c_graph, v_graph)
        cell_output = self.decoder(c_graph)
        potential = cell_output[:, 0:1]  # the scalar potential defined at each cell
        raw_potential = potential.clone()  # Store the raw potential before smoothing
        potential = self.smoother(potential.flatten(), c_graph_save.grad_neighbours)[:, None]

        cell_output_expanded = torch.cat([potential, torch.zeros_like(cell_output[:, 0:1]), cell_output[:, 1:2]], dim=1)
        output_norm = (cell_output_expanded, None, None)
        output = self.normalizer.output(output_norm, inverse=True)

        # create divergence-free velocity field
        cell_velocity = self.divergence_layer(output[0][:, 0], c_graph_save.grad_weights, c_graph_save.grad_neighbours)
        output[0][:, 0:2] = cell_velocity

        if mode == 'train': # for loss
            output = self.normalizer.output(output, inverse=False)

        return {
            'cell_velocity': output[0][:, 0:2],
            'cell_pressure': output[0][:, 2:3],
            'cell_potential': raw_potential,  # Add potential to output
        }

    def loss(self, output, graphs):
        """Extends base loss with potential smoothness regularization"""
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

        # Add Smoothness Regularizer to the Loss Function
        potential = output['cell_potential']
        neighbours = c_graph.grad_neighbours
        neighbour_potentials = potential[neighbours[:, :4]] # Get the potentials of the k-nearest neighbours (k=4)
        laplacian_psi = torch.mean(neighbour_potentials, dim=1) - potential
        potential_smoothness_loss = torch.mean(laplacian_psi**2)
        smoothness_weight = 0.1

        w = self.config.training.loss_weights
        total_loss = w['cell_velocity'] * cell_velocity_loss + w['cell_pressure'] * cell_pressure_loss + smoothness_weight * potential_smoothness_loss
        total_log_loss = torch.mean(torch.log(total_loss))

        return {
            "total_log_loss": total_log_loss,
            "cell_velocity_loss": cell_velocity_loss,
            "cell_pressure_loss": cell_pressure_loss,
            "continuity_loss": continuity
        }

    class SmoothingLayer(torch.nn.Module):
        """Averages potential values over k-nearest neighbours to apply smoothing."""
        def __init__(self, neighbours=3):
            super().__init__()
            self.neighbours = neighbours

        def forward(self, potential, neighbours):
            neighbour_potentials = potential[neighbours[:, :self.neighbours]]
            smoothed_potential = torch.mean(neighbour_potentials, dim=1)

            return smoothed_potential
