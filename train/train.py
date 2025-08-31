from models.linear_model import LinearMomentsTorch
from models.small_deep_moments_mlp import SmallDeepMomentsMLP
from qkal.model import QKAL
from utils.plot import plot_y_space, plot_u_space
from qkal.eval import eval_nll, predict_mean_from_density, rmse, mae, baseline_lr_metrics


from typing import Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import QuantileTransformer
from models.loss import SimpleCoefficientLoss
from utils.config import ReconstructionConfig
from utils.legendre import legendre_targets_from_y


def train_model_from_arrays(
    model,
    X: np.ndarray,
    y: np.ndarray,
    config: ReconstructionConfig) -> Tuple[torch.nn.Module, QuantileTransformer]:

    assert X.ndim == 2 and y.ndim == 1 and len(X) == len(y), "Bad shapes"
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    qt = QuantileTransformer(
        n_quantiles=min(1000, len(y)),
        output_distribution="uniform",
        subsample=10_000,
        random_state=config.seed,
    )
    u = qt.fit_transform(y.reshape(-1, 1)).astype(np.float32).ravel()

    N = len(y)
    idx = np.arange(N)
    rng.shuffle(idx)
    split = int(0.8 * N)
    tr_idx, va_idx = idx[:split], idx[split:]

    X_tr, u_tr = X[tr_idx].astype(np.float32), u[tr_idx]
    X_va, u_va = X[va_idx].astype(np.float32), u[va_idx]

    tr_dl = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(u_tr)),
                       batch_size=config.batch_size, shuffle=True)
    va_dl = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(u_va)),
                       batch_size=config.batch_size, shuffle=False)

    model = model(config).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = SimpleCoefficientLoss(config, legendre_targets_from_y)

    def run_epoch(dl, train: bool) -> float:
        model.train(train)
        tot, n = 0.0, 0
        with torch.set_grad_enabled(train):
            for xb, ub in dl:
                xb = xb.to(dev)
                ub = ub.to(dev).float()

                preds = model(xb)  # (B, K) np.
                out = criterion(preds, ub)
                loss = out[0] if isinstance(out, tuple) else out
                if loss.dim() != 0:
                    loss = loss.mean()

                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

                bs = xb.size(0)
                tot += loss.item() * bs
                n += bs
        return tot / max(n, 1)

    best_val, best_state = float("inf"), None
    for ep in range(1, config.epochs + 1):
        tr = run_epoch(tr_dl, train=True)
        va = run_epoch(va_dl, train=False)
        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"[{ep:02d}/{config.epochs}] train={tr:.5f} val={va:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, qt




if __name__ == "__main__":
    CSV_PATH = r"../model/realtor-data.zip.csv"

    config = ReconstructionConfig()
    config.degree = 3
    config.hidden_dim = 64
    config.batch_size = 4096
    config.grid_size = 512

    X, y = load_your_data(log_price=False)

    if getattr(config, "input_dim", None) != X.shape[1]:
        config.input_dim = X.shape[1]

    models = [
        QKAL,
        LinearMomentsTorch,
        SmallDeepMomentsMLP,
    ]

    #wybierz mdoel
    model, qt = train_model_from_arrays(model, X, y, config)

    # --- EVAL ---
    nll = eval_nll(model, X, y, config)
    yhat = predict_mean_from_density(model, X, y, config)
    print(f"NLL(all): {nll:.4f} | RMSE(all): {rmse(y, yhat):.3f} | MAE(all): {mae(y, yhat):.3f}")
    print(f"[DIAG] y std={y.std():.0f} | yhat std={yhat.std():.0f} | corr(y,yhat)={np.corrcoef(y, yhat)[0, 1]:.3f}")

    print("\n[DIAG] y stats (oryg. skala, całość):")
    q = np.percentile(y, [0, 1, 5, 50, 95, 99, 99.9])
    print(f"min={y.min():.0f}  p1={q[1]:.0f}  p5={q[2]:.0f}  median={q[3]:.0f}  "
          f"p95={q[4]:.0f}  p99={q[5]:.0f}  p99.9={q[6]:.0f}  max={y.max():.0f}")

    print("\n[DIAG] yhat (E[Y|x]) stats (całość):")
    print(f"mean={yhat.mean():.0f}  std={yhat.std():.0f}  min={yhat.min():.0f}  max={yhat.max():.0f}")

    # Baseline
    base = baseline_lr_metrics(X, y, X, y)
    print(f"\n[Baseline LR] NLL(all): {base['nll']:.4f} | RMSE(all): {base['rmse']:.0f} | MAE(all): {base['mae']:.0f}")

    y_of_u, f_y_batch, u_grid, rho_u_batch, dy_du, p_marg_y = density_from_qkal(
        model, X, y, config
    )
    plot_u_space(
        u_grid = u_grid,
        rho_u_batch = rho_u_batch,
        n_show=config.n_of_reconstructions
    )

    plot_y_space(
        y_of_u = y_of_u,
        f_y_batch = f_y_batch,
        y_data = y,
        p_marg_y = p_marg_y,
        n_show=config.n_of_reconstructions
    )
