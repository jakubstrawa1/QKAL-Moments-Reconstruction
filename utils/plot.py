import numpy as np
import matplotlib.pyplot as plt
import torch

EPS = 1e-12

# ------------- helpers -------------
def _to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)

def _cumtrapz_batch(y_batch: np.ndarray, x: np.ndarray) -> np.ndarray:

    y_batch = np.asarray(y_batch, dtype=float)
    x = np.asarray(x, dtype=float)

    dx = np.diff(x)                                   # (N-1,)
    avg = 0.5 * (y_batch[:, :-1] + y_batch[:, 1:])    # (B, N-1)
    part = np.cumsum(avg * dx[None, :], axis=1)       # (B, N-1)
    cdf = np.concatenate([np.zeros((y_batch.shape[0], 1)), part], axis=1)  # (B, N)

    denom = np.clip(cdf[:, -1:], EPS, None)
    cdf = cdf / denom
    cdf[:, -1] = 1.0
    return cdf

# ------------- u-space -------------
def plot_u_space(u_grid, rho_u_batch, n_show=10, title="Model density in u-space ρ(u|x)"):
    u = _to_numpy(u_grid).ravel()
    rho = _to_numpy(rho_u_batch)
    plt.figure(figsize=(8, 4))
    for i in range(min(n_show, rho.shape[0])):
        plt.plot(u, rho[i], label=f"ρ(u|x[{i}])")
    plt.axhline(1.0, linestyle="--", label="uniform ref")
    plt.xlabel("u ∈ (0,1)")
    plt.ylabel("ρ(u|x)")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

# ------------- y-space -------------
def plot_y_space(y_of_u, f_y_batch, y_data=None, p_marg_y=None, n_show=10,
                 title="Conditional density in y-space f(y|x)"):
    y = _to_numpy(y_of_u).ravel()
    f = _to_numpy(f_y_batch)
    plt.figure(figsize=(8, 4))
    for i in range(min(n_show, f.shape[0])):
        plt.plot(y, f[i], label=f"f(y|x[{i}])")
    if p_marg_y is not None:
        plt.plot(y, _to_numpy(p_marg_y), "k--", lw=2, label="marginal KDE")
    plt.xlabel("y")
    plt.ylabel("density")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

# ------------- PDF area -------------
def plot_pdf_normalization(y_grid_t, f_batch_t, title="PDF area per sample"):
    y = _to_numpy(y_grid_t)
    f = _to_numpy(f_batch_t)
    area = np.trapz(f, x=y, axis=1)
    plt.figure(figsize=(6, 3.5))
    plt.hist(area, bins=30, alpha=0.9)
    plt.axvline(1.0, linestyle="--")
    plt.xlabel("∫ f(y|x) dy")
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ------------- NLL histogram -------------
def plot_nll_hist(nll_per_sample, title="Per-sample NLL"):
    nll_per_sample = np.asarray(nll_per_sample, dtype=float)
    plt.figure(figsize=(6, 3.5))
    plt.hist(nll_per_sample, bins=40, alpha=0.9)
    p90 = np.percentile(nll_per_sample, 90)
    plt.axvline(p90, linestyle="--", label=f"p90={p90:.3f}")
    plt.xlabel("NLL_i = -log f(y_i|x_i)")
    plt.ylabel("count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------- PIT -------------
def plot_pit_hist(y_grid_t, f_y_batch_t, y_true, bins: int = 20, title: str = "PIT histogram"):

    y = _to_numpy(y_grid_t).ravel()
    f = _to_numpy(f_y_batch_t)
    yt = _to_numpy(y_true).ravel()


    area = np.trapz(f, x=y, axis=1)
    f = f / np.clip(area[:, None], EPS, None)


    cdf = _cumtrapz_batch(f, y)

    # PIT: F(y_true)
    B = min(len(yt), f.shape[0])
    pit = np.empty(B, dtype=float)
    for i in range(B):
        pit[i] = np.interp(yt[i], y, cdf[i], left=0.0, right=1.0)

    plt.figure(figsize=(6, 4))
    plt.hist(pit, bins=bins, range=(0.0, 1.0), density=True, alpha=0.85)
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("PIT")
    plt.ylabel("density")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ------------- coverage -------------
def plot_interval_coverage(y_grid_t, f_y_batch_t, y_true, qs=(0.05, 0.95), title="Interval coverage"):
    """
    Sprawdza pokrycie nominalnego przedziału kwantylowego [q_low, q_high].
    """
    q_low, q_high = float(qs[0]), float(qs[1])

    y = _to_numpy(y_grid_t).ravel()
    f = _to_numpy(f_y_batch_t)
    yt = _to_numpy(y_true).ravel()


    area = np.trapz(f, x=y, axis=1)
    f = f / np.clip(area[:, None], EPS, None)


    cdf = _cumtrapz_batch(f, y)

    B = min(len(yt), f.shape[0])
    low = np.empty(B, dtype=float)
    high = np.empty(B, dtype=float)


    for i in range(B):
        low[i]  = np.interp(q_low,  cdf[i], y, left=y[0], right=y[-1])
        high[i] = np.interp(q_high, cdf[i], y, left=y[0], right=y[-1])

    covered = ((yt[:B] >= low) & (yt[:B] <= high)).mean()
    nominal = q_high - q_low

    plt.figure(figsize=(5, 4))
    plt.bar([0], [covered], width=0.5, label="observed")
    plt.axhline(nominal, color="k", linestyle="--", label=f"nominal ({nominal:.0%})")
    plt.ylim(0, 1)
    plt.xticks([])
    plt.ylabel("coverage")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------- pred vs true -------------
def plot_pred_vs_true(y_true, y_pred, title="E[Y|x] vs Y"):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(yt, yp, s=12, alpha=0.6)
    plt.plot([lo, hi], [lo, hi], "--")
    plt.xlabel("True Y")
    plt.ylabel("Predicted E[Y|x]")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ------------- residuals -------------
def plot_residuals(y_true, y_pred, title="Residuals vs. prediction"):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    resid = yt - yp
    plt.figure(figsize=(6, 3.5))
    plt.scatter(yp, resid, s=12, alpha=0.6)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Predicted E[Y|x]")
    plt.ylabel("Residual (Y - E[Y|x])")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ------------- z-space (log1p) -------------
def plot_z_space(z_of_u_t, f_z_batch_t, z_data_t=None, p_marg_z_t=None, n_show=10,
                 title="Conditional density in z-space f(z|x)"):
    z = _to_numpy(z_of_u_t)
    fz = _to_numpy(f_z_batch_t)
    plt.figure(figsize=(8, 4))
    for i in range(min(n_show, fz.shape[0])):
        plt.plot(z, fz[i], label=f"f(z|x[{i}])")
    if p_marg_z_t is not None:
        plt.plot(z, _to_numpy(p_marg_z_t), "k--", lw=2, label="marginal KDE (z)")
    plt.xlabel("z = log(1+y)")
    plt.ylabel("density in z")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

# ------------- sharpness vs error -------------
def plot_sharpness_vs_error(y_grid_t, f_batch_t, y_true, y_pred, title="Sharpness vs Error"):

    y = _to_numpy(y_grid_t).ravel()          # (N,)
    f = _to_numpy(f_batch_t)                  # (B, N)


    area = np.trapz(f, x=y, axis=1)          # (B,)
    f = f / np.clip(area.reshape(-1, 1), EPS, None)

    dy = np.gradient(y)                       # (N,)
    ent = -np.sum(f * np.log(np.clip(f, EPS, None)) * dy[None, :], axis=1)  # (B,)

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    B = min(len(ent), len(yt), len(yp))
    err = np.abs(yt[:B] - yp[:B])

    plt.figure(figsize=(6, 3.5))
    plt.scatter(ent[:B], err, s=12, alpha=0.6)
    plt.xlabel("sharpness ~ -∫ f log f dy  (mniejsze = ostrzejsze)")
    plt.ylabel("|Y - E[Y|x]|")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_kde_bandwidth_sweep(bw_to_pdf_list, sample_idx=0, title="KDE bandwidth sweep (one x)"):
    plt.figure(figsize=(8, 4))
    for (bw, (y_grid_t, f_batch_t)) in bw_to_pdf_list:
        y = _to_numpy(y_grid_t).ravel()
        f = _to_numpy(f_batch_t)[sample_idx]
        area = np.trapz(f, x=y)
        f = f / max(area, EPS)
        plt.plot(y, f, label=f"BW={bw:g}")
    plt.xlabel("y")
    plt.ylabel("density")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
