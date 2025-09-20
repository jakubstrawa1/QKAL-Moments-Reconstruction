from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from models.density import density_from_model
from utils.legendre import compute_scaled_legendre_polynomials
import torch.nn.functional as F

EPS = 1e-12

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def mae(a, b):  return float(np.mean(np.abs(a - b)))

def _transform_y(y: np.ndarray, space: str) -> np.ndarray:
    if space == "raw":
        return y.astype(np.float32)
    elif space == "log1p":
        if np.any(y < -1):
            raise ValueError("space='log1p' wymaga y >= -1.")
        return np.log1p(y).astype(np.float32)
    else:
        raise ValueError(f"Unknown space='{space}', use 'raw' or 'log1p'.")

def _inv_on_mean(s_grid: np.ndarray, fs: np.ndarray, space: str) -> np.ndarray:
    if space == "raw":
        Ey = np.trapz(s_grid * fs, x=s_grid, axis=1)
    elif space == "log1p":
        Ey = np.trapz((np.expm1(s_grid)) * fs, x=s_grid, axis=1)
    else:
        raise ValueError
    return Ey


# === NLL evaluation ===
@torch.no_grad()
def evaluate_nll_legendre(
    model,
    X_te: np.ndarray,
    y_te: np.ndarray,
    qt,
    degree: int,
    device: str = "cpu",
    grid_size: int = 512,
    batch_size: int = 4096,
    eps: float = 1e-12,
) -> float:
    """
    Computes mean NLL on test set:
      p(y|x) = p(u|x) * du/dy  with  u = qt.transform(y)
    where p(u|x) is reconstructed from Legendre coefficients predicted by the model.

    Reconstruction:
      raw(u|x) = sum_k a_k(x) φ_k(u)
      p(u|x)   = softplus(raw)/Z(x),   Z(x)=∫ softplus(raw(u'|x)) du'  (numerical on [0,1])
    """
    model.eval()
    device = torch.device(device)

    coeffs_list = []
    for start in range(0, len(X_te), batch_size):
        xb = torch.from_numpy(X_te[start:start+batch_size]).to(device=device, dtype=torch.float32)
        coeffs_list.append(model(xb).detach().cpu())
    coeffs = torch.cat(coeffs_list, dim=0)              # (N, degree+1)
    assert coeffs.shape[1] == degree + 1, f"Got {coeffs.shape[1]} coeffs; expected {degree+1}"

    u_te = qt.transform(y_te.reshape(-1, 1)).astype(np.float32).ravel()     # (N,)
    dy = max(1e-4 * np.std(y_te) if np.std(y_te) > 0 else 1e-4, 1e-6)
    u_plus  = qt.transform((y_te + dy).reshape(-1, 1)).ravel()
    u_minus = qt.transform((y_te - dy).reshape(-1, 1)).ravel()
    du_dy = (u_plus - u_minus) / (2.0 * dy)
    du_dy = np.clip(du_dy, eps, None)                                       # avoid zeros
    u_te_t = torch.from_numpy(u_te).to(dtype=torch.float32)                 # stays on CPU (small ops)

    grid = torch.linspace(0.0, 1.0, steps=grid_size, dtype=torch.float32)   # (G,)
    Phi_grid = compute_scaled_legendre_polynomials(grid.unsqueeze(0), degree).squeeze(0)  # (G, K)

    log_py = []
    G = grid_size
    dx = 1.0 / (G - 1)

    for start in range(0, len(X_te), batch_size):
        end = start + batch_size
        a = coeffs[start:end]                             # (B, K)

        # raw on grid: (B,G) = (B,K) @ (K,G)
        raw_grid = torch.matmul(a, Phi_grid.T)            # (B, G)
        pos_grid = F.softplus(raw_grid)                   # positive
        # trapezoidal normalization
        Z = (pos_grid[:, 0] + pos_grid[:, -1]) * 0.5 + pos_grid[:, 1:-1].sum(dim=1)
        Z = Z * dx                                        # (B,)
        Z = torch.clamp(Z, min=eps)

        # evaluate at the sample's u_i
        u_batch = u_te_t[start:end]                       # (B,)
        Phi_u = compute_scaled_legendre_polynomials(u_batch.unsqueeze(1), degree).squeeze(1)  # (B,K)
        raw_u = (a * Phi_u).sum(dim=1)                    # (B,)
        pos_u = F.softplus(raw_u)                         # (B,)
        p_u = pos_u / Z                                   # (B,)

        # Jacobian to y-space
        p_y = p_u * torch.from_numpy(du_dy[start:end]).to(dtype=torch.float32)  # (B,)
        p_y = torch.clamp(p_y, min=eps)
        log_py.append(torch.log(p_y))

    log_py = torch.cat(log_py, dim=0)                     # (N,)
    nll = -float(log_py.mean().item())
    return nll


@torch.no_grad()
def predict_mean_from_density(
    model,
    X: np.ndarray,
    y_like: np.ndarray,
    config,
    *,
    batch_size: int = 2048,
    space: str = "raw",
    y_ref_raw: np.ndarray | None = None,
    qt=None
) -> np.ndarray:
    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        y_eval = _transform_y(y_like, space)
        if y_ref_raw is None:
            y_ref_raw = y_like
        y_ref_t_raw = torch.from_numpy(y_ref_raw.astype(np.float32)).to(device)

        dl = DataLoader(
            TensorDataset(torch.from_numpy(X.astype(np.float32)),
                          torch.from_numpy(y_eval.astype(np.float32))),
            batch_size=batch_size, shuffle=False
        )
        outs = []
        for xb, yb in dl:
            s_grid_t, f_s_batch_t, *_ = density_from_model(
                model, xb.to(device), y_ref_t_raw, config, qt=qt, space=space
            )
            s_grid = s_grid_t.detach().cpu().numpy()
            fs     = f_s_batch_t.detach().cpu().numpy()
            area = np.trapz(fs, x=s_grid, axis=1)
            fs = fs / np.clip(area[:, None], EPS, None)
            Ey = _inv_on_mean(s_grid, fs, space=space)
            outs.append(Ey)
        return np.concatenate(outs, axis=0)
    finally:
        if was_training:
            model.train()

def baseline_lr_metrics(X_tr, y_tr, X_te, y_te, space: str = "raw"):
    lr = LinearRegression().fit(X_tr, y_tr)
    mu_tr = lr.predict(X_tr)
    mu_te = lr.predict(X_te)

    if space == "raw":
        resid_tr = y_tr - mu_tr
        sigma2 = float(np.var(resid_tr, ddof=1))
        sigma2 = max(sigma2, EPS)
        nll_te = float(np.mean(0.5*np.log(2*np.pi*sigma2) + (y_te - mu_te)**2/(2*sigma2)))
    elif space == "log1p":
        z_tr = np.log1p(y_tr); z_mu_tr = np.log1p(np.clip(mu_tr, 0, None))
        z_te = np.log1p(y_te); z_mu_te = np.log1p(np.clip(mu_te, 0, None))
        sigma2 = float(np.var(z_tr - z_mu_tr, ddof=1))
        sigma2 = max(sigma2, EPS)
        nll_te = float(np.mean(0.5*np.log(2*np.pi*sigma2) + (z_te - z_mu_te)**2/(2*sigma2)))
    else:
        raise ValueError("space must be 'raw' or 'log1p'")

    return {"rmse": rmse(y_te, mu_te), "mae": mae(y_te, mu_te), "nll": nll_te, "model": lr}
