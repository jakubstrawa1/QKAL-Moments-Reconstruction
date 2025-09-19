from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from models.density import density_from_model

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

@torch.no_grad()
def eval_nll(
    model,
    X: np.ndarray,
    y: np.ndarray,
    config,
    *,
    batch_size: int = 2048,
    space: str = "log1p",
    y_ref_raw: np.ndarray | None = None,
    qt=None
) -> float:
    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        y_eval = _transform_y(y, space)

        if y_ref_raw is None:
            y_ref_raw = y
        y_ref_t_raw = torch.from_numpy(y_ref_raw.astype(np.float32)).to(device)

        dl = DataLoader(
            TensorDataset(torch.from_numpy(X.astype(np.float32)),
                          torch.from_numpy(y_eval.astype(np.float32))),
            batch_size=batch_size, shuffle=False
        )

        nll_sum, n_obs = 0.0, 0
        for xb, yb in dl:
            xb = xb.to(device)
            s_grid_t, f_s_batch_t, *_ = density_from_model(
                model, xb, y_ref_t_raw, config, qt=qt, space=space
            )
            s_grid = s_grid_t.detach().cpu().numpy()
            fs     = f_s_batch_t.detach().cpu().numpy()
            if fs.ndim == 1:
                fs = fs[None, :]
                y_np = yb.cpu().numpy().reshape(1)
            else:
                y_np = yb.cpu().numpy()
            area = np.trapz(fs, x=s_grid, axis=1)
            fs = fs / np.clip(area[:, None], EPS, None)
            for i in range(len(y_np)):
                fi = np.interp(y_np[i], s_grid, fs[i], left=EPS, right=EPS)
                nll_sum += -np.log(max(fi, EPS))
            n_obs += len(y_np)

        return nll_sum / max(n_obs, 1)
    finally:
        if was_training:
            model.train()

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
