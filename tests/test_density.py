import numpy as np
import torch
import pytest

from qkal.density import density_from_qkal, empirical_cdf, inverse_cdf
from utils.config import QKALReconstructionConfig, CalibrationMode
import torch.nn as nn


class DummyModel(nn.Module):

    def __init__(self, rho_scale=1.0, device="cpu", dtype=torch.float32):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0, dtype=dtype, device=device))
        self.rho_scale = float(rho_scale)

    @torch.no_grad()
    def predict_density_u(self, x: torch.Tensor, nn_grid: int, mode):
        B = x.shape[0]
        device, dtype = x.device, x.dtype
        k = torch.arange(1, nn_grid + 1, device=device, dtype=dtype)
        u_grid = (k - 0.5) / nn_grid
        rho = torch.full((B, nn_grid), self.rho_scale,
                         device=device, dtype=dtype)
        return rho, u_grid


def _cfg(grid=64, norm=True, bw=None):
    cfg = QKALReconstructionConfig()
    cfg.grid_size = grid
    cfg.kde_bandwidth = bw
    cfg.normalize_density = norm
    cfg.calibration_mode = CalibrationMode.CALIBRATED_SOFTPLUS
    return cfg


def test_density_shapes_and_normalization_true():
    torch.manual_seed(0)
    B, D, N = 7, 5, 300
    X = torch.randn(B, D).numpy().astype(np.float32)
    # dane y ~ N(0,1)
    y = torch.randn(N).numpy().astype(np.float32)

    model = DummyModel(rho_scale=1.0)
    cfg = _cfg(grid=64, norm=True)

    y_of_u, f_y_batch, u_grid, rho_u_batch, dy_du, p_marg_y = density_from_qkal(
        model, X, y, cfg
    )

    assert y_of_u.ndim == 1 and y_of_u.numel() == cfg.grid_size
    assert f_y_batch.shape == (B, cfg.grid_size)
    assert rho_u_batch.shape == (B, cfg.grid_size)
    assert dy_du.shape == (cfg.grid_size,)
    assert p_marg_y.shape == (cfg.grid_size,)

    yu = y_of_u.detach().cpu().numpy()
    assert np.all(np.diff(yu) >= -1e-7)
    assert torch.all(dy_du > 0)

    du = 1.0 / cfg.grid_size
    inte = torch.sum(f_y_batch * dy_du.unsqueeze(0), dim=1) * du
    assert torch.allclose(inte, torch.ones_like(inte), atol=5e-2)


def test_density_no_normalize_scales_with_rho():
    torch.manual_seed(0)
    B, D, N = 3, 4, 200
    X = torch.randn(B, D).numpy().astype(np.float32)
    y = (torch.randn(N) * 2.0 + 1.5).numpy().astype(np.float32)

    model = DummyModel(rho_scale=2.0)
    cfg = _cfg(grid=64, norm=False)

    y_of_u, f_y_batch, u_grid, rho_u_batch, dy_du, p_marg_y = density_from_qkal(
        model, X, y, cfg
    )
    du = 1.0 / cfg.grid_size
    inte = torch.sum(f_y_batch * dy_du.unsqueeze(0), dim=1) * du
    assert torch.all((inte > 1.5) & (inte < 2.5)), f"Integrals out of expected range: {inte}"


def test_empirical_and_inverse_cdf_consistency():

    torch.manual_seed(0)
    y = torch.randn(500).sort().values    # dane 1D
    y_min, y_max = y.min().item(), y.max().item()
    y_grid = torch.linspace(y_min - 0.1, y_max + 0.1, 200)
    cdf_vals = empirical_cdf(y, y_grid)

    u = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32)
    y_q = inverse_cdf(u, y_grid, cdf_vals)
    cdf_at_yq = empirical_cdf(y, y_q)
    assert torch.allclose(cdf_at_yq, u, atol=0.05), f"{cdf_at_yq} vs {u}"
