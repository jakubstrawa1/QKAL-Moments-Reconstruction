from __future__ import annotations
import argparse, os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from utils.data import engineer_housing_features
from utils.config import ReconstructionConfig, CalibrationMode
from train.moments_trainer import make_loaders, train_one_model
from eval.eval import eval_nll, predict_mean_from_density, rmse, mae
from models.density import density_from_model
from utils.plot import plot_u_space, plot_y_space

from models.qkal import QKAL
from models.linear_model import LinearMomentsTorch
from models.small_deep_moments_mlp import SmallDeepMomentsMLP


def plot_histories(all_results):
    for deg, results in all_results.items():  # deg -> {model_name: {...}}
        plt.figure(figsize=(10, 6))
        for model_name, data in results.items():
            tr = data["history"]["train"]
            va = data["history"]["val"]
            xs = range(1, len(tr) + 1)
            plt.plot(xs, tr, label=f"{model_name} - train")
            plt.plot(xs, va, "--", label=f"{model_name} - val")
        plt.xlabel("Epoch"); plt.ylabel("Loss (MSE on Legendre coeffs)")
        plt.title(f"Training & Validation Loss (degree={deg})")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.show()


def evaluate_full(
    model,
    X_te: np.ndarray,
    y_te: np.ndarray,
    cfg: ReconstructionConfig,
    *,
    y_ref_raw: np.ndarray,
    do_plots: bool = False,
    eval_spaces = ("raw", "log1p"),
):
    X_te = np.asarray(X_te, dtype=np.float32)
    y_te = np.asarray(y_te, dtype=np.float32)

    # NLL (siatka/marginala zbudowane na train+val)
    nlls = {}
    for sp in eval_spaces:
        y_ref_for_space = y_ref_raw if sp == "raw" else np.log1p(y_ref_raw)
        nlls[sp] = float(eval_nll(model, X_te, y_te, cfg, space=sp, y_ref_for_space=y_ref_for_space))

    yhat = predict_mean_from_density(model, X_te, y_te, cfg, space="raw", y_ref_for_space=y_ref_raw)

    out = {
        "NLL_raw": nlls.get("raw", np.nan),
        "NLL_log1p": nlls.get("log1p", np.nan),
        "RMSE": rmse(y_te, yhat),
        "MAE": mae(y_te, yhat),
        "corr": float(np.corrcoef(y_te, yhat)[0, 1]),
        "mean_y": float(y_te.mean()),
        "mean_yhat": float(yhat.mean()),
    }

    if do_plots:
        N = min(256, len(y_te))
        y_of_u, f_y_batch, u_grid, rho_u_batch, _, p_marg_y = density_from_model(
            model, X_te[:N], y_ref_raw, cfg
        )
        plot_u_space(u_grid, rho_u_batch, n_show=min(8, N))
        plot_y_space(y_of_u, f_y_batch, y_data=y_te[:N], p_marg_y=p_marg_y, n_show=min(8, N))

    return out


def build_parser():
    ap = argparse.ArgumentParser("QKAL/MLP/LINEAR moments — housing (+ NLL raw/log1p, RMSE/MAE/corr)")
    ap.add_argument("--csv", default="h_prices.csv")
    ap.add_argument("--target", default="House_Price")
    ap.add_argument("--scale_x", action="store_true")
    ap.add_argument("--degrees", default="3", help="np. '3,4,5'")
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plots", action="store_true", help="po ewaluacji narysuj gęstości (holdout)")
    ap.add_argument("--with_linear", action="store_true", help="dołącz baseline Linear")
    return ap


def main():
    args = build_parser().parse_args()
    np.random.seed(args.seed)

    df = pd.read_csv(args.csv)
    fe = engineer_housing_features(df, target_col=args.target, scale=False, add_log_target=False)
    X_df = fe["X"]; y_sr = fe["y"]

    X = X_df.to_numpy(dtype=np.float32)
    y = y_sr.to_numpy(dtype=np.float32)

    if args.scale_x:
        X = StandardScaler().fit_transform(X).astype(np.float32)

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(y)); rng.shuffle(idx)
    n_test = int(args.test_ratio * len(y))
    te_idx = idx[:n_test]
    tv_idx = idx[n_test:]

    X_te, y_te = X[te_idx], y[te_idx]
    X_tv, y_tv = X[tv_idx], y[tv_idx]

    tr_dl, va_dl = make_loaders(X_tv, y_tv, batch_size=args.batch_size, val_ratio=args.val_ratio)

    degrees = [int(d.strip()) for d in args.degrees.split(",") if d.strip()]
    all_results = {}
    best_by_model = {}
    best_models = {}

    for deg in degrees:
        print(f"\n=== DEGREE = {deg} ===")
        cfg = ReconstructionConfig(
            degree=deg,
            input_dim=X.shape[1],
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=3e-4,
            weight_decay=1e-3,
        )
        cfg.calibration_mode = CalibrationMode.CALIBRATED_SOFTPLUS
        cfg.grid_size = getattr(cfg, "grid_size", 128)

        models = {
            "MLP":  SmallDeepMomentsMLP(cfg),
            "QKAL": QKAL(cfg),
            "LINEAR": LinearMomentsTorch(cfg)
        }

        results_deg = {}
        for name, model in models.items():
            print(f"\n--- Training {name} ---")
            res = train_one_model(model, tr_dl, va_dl, cfg, loss="mse")
            results_deg[name] = res

            val = res["best_val"]
            if (name not in best_by_model) or (val < best_by_model[name][1]):
                best_by_model[name] = (deg, val)
                best_models[name] = (res["model"], cfg)  # wgrane best wagi

        all_results[deg] = results_deg

    print("\nBest degree per model (by val loss):")
    for name, (deg, val) in best_by_model.items():
        print(f"{name}: degree={deg}, val_loss={val:.6f}")

    rows = []
    for name, (model, cfg_used) in best_models.items():
        print(f"\n>>> TEST EVAL: {name.lower()} (best degree={cfg_used.degree}, val={best_by_model[name][1]:.6f})")
        metrics = evaluate_full(
            model, X_te, y_te, cfg_used,
            y_ref_raw=y_tv,
            do_plots=args.plots,
            eval_spaces=("raw", "log1p"),
        )
        rows.append({"model": name, **metrics})

    if rows:
        df_sum = pd.DataFrame(rows)
        print("\n=== SUMMARY (holdout / test) ===")
        cols = ["model","NLL_raw","NLL_log1p","RMSE","MAE","corr","mean_y","mean_yhat"]
        print(df_sum[cols].to_string(index=False))
        os.makedirs("runs", exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        df_sum.to_csv(f"runs/housing_summary_{ts}.csv", index=False)

    plot_histories(all_results)


if __name__ == "__main__":
    main()
