# main.py
from __future__ import annotations
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

from utils.config import ReconstructionConfig, CalibrationMode
from utils.data import load_realtor_csv
from train.train import train_model_from_arrays
from eval.eval import eval_nll, predict_mean_from_density, rmse, mae
from models.density import density_from_model
from utils.plot import plot_u_space, plot_y_space

from models.qkal import QKAL
from models.linear_model import LinearMomentsTorch
from models.small_deep_moments_mlp import SmallDeepMomentsMLP

MODEL_REGISTRY = {
    "qkal": QKAL,
    "linear": LinearMomentsTorch,
    "mlp": SmallDeepMomentsMLP,
}

def run_one(model_cls, X: np.ndarray, y: np.ndarray, cfg: ReconstructionConfig, seed=42, do_plots=False):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y)); rng.shuffle(idx)
    split = int(0.8 * len(y))
    tr, te = idx[:split], idx[split:]
    X_tr, y_tr = X[tr], y[tr]
    X_te, y_te = X[te], y[te]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    cfg2 = ReconstructionConfig(**vars(cfg))
    cfg2.input_dim = X_tr.shape[1]

    # Stabilniejsza kalibracja dla prostszych modeli bo dziwnie działały
    if model_cls in (LinearMomentsTorch, SmallDeepMomentsMLP):
        cfg2.calibration_mode = CalibrationMode.SOFTPLUS
        cfg2.degree = min(cfg2.degree, 3)
        if model_cls is LinearMomentsTorch:
            cfg2.calibration_mode = CalibrationMode.SOFTPLUS
            cfg2.degree = min(cfg2.degree, 2)
            cfg2.learning_rate = min(cfg2.learning_rate, 3e-4)
            cfg2.weight_decay = max(cfg2.weight_decay, 1e-3)
            cfg2.epochs = min(cfg2.epochs, 30)

    model, _ = train_model_from_arrays(model_cls, X_tr, y_tr, cfg2)

    # --- EVAL ---
    nll  = eval_nll(model, X_te, y_te, cfg2)
    yhat = predict_mean_from_density(model, X_te, y_te, cfg2)
    r = {
        "model": model_cls.__name__,
        "NLL": float(nll),
        "RMSE": rmse(y_te, yhat),
        "MAE": mae(y_te, yhat),
        "corr": float(np.corrcoef(y_te, yhat)[0,1]),
        "mean_y": float(y_te.mean()),
        "mean_yhat": float(yhat.mean()),
    }
    print(f"NLL={r['NLL']:.4f} | RMSE={r['RMSE']:.0f} | MAE={r['MAE']:.0f} | corr={r['corr']:.3f}")

    if do_plots:
        N = min(256, len(y_te))
        y_of_u, f_y_batch, u_grid, rho_u_batch, dy_du, p_marg_y = density_from_model(model, X_te[:N], y_te[:N], cfg2)
        plot_u_space(u_grid, rho_u_batch, n_show=min(8, N))
        plot_y_space(y_of_u, f_y_batch, y_data=y_te[:N], p_marg_y=p_marg_y, n_show=min(8, N))

    return r

def main():
    ap = argparse.ArgumentParser(description="QKAL vs Linear vs MLP on realtor-data.zip.csv")
    ap.add_argument("--csv", default="realtor-data.zip.csv", help="ścieżka do CSV z kolumną price")
    ap.add_argument("--models", default="qkal,linear,mlp", help="lista: qkal,linear,mlp (np. 'qkal,mlp')")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plots", action="store_true", help="narysuj gęstości na próbce holdoutu")

    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--grid_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=(4096 if torch.cuda.is_available() else 1024))
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    #dane
    X, y = load_realtor_csv(args.csv, numeric_only=False, log_price=False)

    cfg = ReconstructionConfig(
        degree=args.degree,
        hidden_dim=args.hidden_dim,
        grid_size=args.grid_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    #modele
    names = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"Modele: {names}\nDługość zbioru: {len(y)}  |  input_dim (po featurach) = {X.shape[1]}")
    results = []
    for name in names:
        if name not in MODEL_REGISTRY:
            raise ValueError(f"Nieznany model '{name}'. Do wyboru: {list(MODEL_REGISTRY.keys())}")
        print(f"\n=== {name.upper()} ===")
        res = run_one(MODEL_REGISTRY[name], X, y, cfg, seed=args.seed, do_plots=args.plots)
        results.append(res)

    try:
        import pandas as pd
        df = pd.DataFrame(results)
        print("\n=== SUMMARY ===")
        print(df[["model","NLL","RMSE","MAE","corr","mean_y","mean_yhat"]].to_string(index=False))
    except Exception:
        pass

if __name__ == "__main__":
    main()
