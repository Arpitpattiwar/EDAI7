"""
Physics-Informed Neural Network (PINN) Architecture for Urban LST Prediction.
Embeds continuous spatial coordinates (x, y), time (t), and multi-spectral/environmental features.
Uses smooth SiLU activations to enable clean higher-order automatic differentiation for PDE residuals.
"""

import torch
import torch.nn as nn


class MumbaiPINN(nn.Module):
    """
    PINN architecture:
    Inputs:
      - coords: Tensor of shape (batch, 3) representing [x_norm, y_norm, t_norm]
      - features: Tensor of shape (batch, n_features)
    Output:
      - pred_lst: Tensor of shape (batch, 1) representing predicted LST in °C
    """
    def __init__(self, n_features=15, hidden_dim=128, n_layers=4):
        super(MumbaiPINN, self).__init__()
        self.n_features = n_features
        total_input_dim = 3 + n_features  # [x, y, t] + features

        layers = []
        layers.append(nn.Linear(total_input_dim, hidden_dim))
        layers.append(nn.SiLU())  # Smooth activation with continuous second derivatives

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

        # Weight initialization for stable autograd gradients
        self._init_weights()

    def _init_weights(self):
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, coords, features):
        """
        coords: (batch, 3) with [x, y, t]
        features: (batch, n_features)
        """
        inputs = torch.cat([coords, features], dim=-1)
        out = self.network(inputs)
        return out
