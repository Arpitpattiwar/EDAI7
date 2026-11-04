"""
Physics-Informed Loss Module for Mumbai Advection-Diffusion Heat Transfer.
Implements automatic differentiation for spatial and temporal gradients,
directional sea-breeze advection, and learnable PDE parameters (diffusivity alpha, wind decay kd).
"""

import torch
import torch.nn as nn


class AdvectionDiffusionPhysicsLoss(nn.Module):
    """
    Computes the residual of the 2D advection-diffusion heat equation:
      Residual = dT/dt - alpha * (d2T/dx2 + d2T/dy2) + (u_eff * dT/dx + v_eff * dT/dy) - S + lambda_exch * (T - T_air)

    where:
      u_eff = u_wind * exp(-dist_to_sea / kd)
      v_eff = v_wind * exp(-dist_to_sea / kd)
    """
    def __init__(self, alpha_init=0.05, kd_init=8.0, lambda_exch=0.15, stage="stage3"):
        super(AdvectionDiffusionPhysicsLoss, self).__init__()
        self.stage = stage

        # Learnable physical parameters (Inverse PINN / Parameter Discovery)
        # alpha: effective thermal diffusivity
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(float(alpha_init), dtype=torch.float32)))
        # kd: sea-breeze penetration decay length scale (km)
        self.log_kd = nn.Parameter(torch.log(torch.tensor(float(kd_init), dtype=torch.float32)))

        self.lambda_exch = lambda_exch

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    @property
    def kd(self):
        return torch.exp(self.log_kd)

    def compute_residual(self, model, coords, features, raw_feature_dict):
        """
        Evaluates the PDE residual using autograd.
        coords: Tensor of shape (batch, 3) [x, y, t], requires_grad=True
        features: Standardized features tensor (batch, n_features)
        raw_feature_dict: Dictionary containing unscaled physical variables:
          - dist_to_sea_km, u_wind, v_wind, air_temp_c, solar_radiation, albedo, ndbi
        """
        if not coords.requires_grad:
            coords = coords.clone().detach().requires_grad_(True)

        T = model(coords, features)

        grad_outputs = torch.ones_like(T)
        grads = torch.autograd.grad(
            outputs=T,
            inputs=coords,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        dT_dx = grads[:, 0:1]
        dT_dy = grads[:, 1:2]
        dT_dt = grads[:, 2:3]

        d2T_dx2 = torch.autograd.grad(
            outputs=dT_dx,
            inputs=coords,
            grad_outputs=torch.ones_like(dT_dx),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0][:, 0:1]

        d2T_dy2 = torch.autograd.grad(
            outputs=dT_dy,
            inputs=coords,
            grad_outputs=torch.ones_like(dT_dy),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0][:, 1:2]

        laplacian = d2T_dx2 + d2T_dy2

        dist_km = raw_feature_dict["dist_to_sea_km"].unsqueeze(1)
        u_wind = raw_feature_dict["u_wind"].unsqueeze(1)
        v_wind = raw_feature_dict["v_wind"].unsqueeze(1)
        air_temp = raw_feature_dict["air_temp_c"].unsqueeze(1)
        solar = raw_feature_dict["solar_radiation"].unsqueeze(1)
        albedo = raw_feature_dict["albedo"].unsqueeze(1)
        ndbi = raw_feature_dict["ndbi"].unsqueeze(1)

        S_solar = solar * (1.0 - albedo) / 35.0
        S_urban = 3.5 * torch.clamp(ndbi, min=0.0)
        S_total = S_solar + S_urban

        exchange_term = self.lambda_exch * (T - air_temp)

        if self.stage == "stage1":
            residual = torch.zeros_like(T)
        elif self.stage == "stage2":
            diffusion_term = self.alpha * laplacian
            residual = dT_dt - diffusion_term + exchange_term - S_total
        else:
            decay_factor = torch.exp(-torch.clamp(dist_km, min=0.0) / (self.kd + 1e-6))
            u_eff = u_wind * decay_factor
            v_eff = v_wind * decay_factor

            advection_term = u_eff * dT_dx + v_eff * dT_dy
            diffusion_term = self.alpha * laplacian

            residual = dT_dt - diffusion_term + advection_term + exchange_term - S_total

        return residual, T
