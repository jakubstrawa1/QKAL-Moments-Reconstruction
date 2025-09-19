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
from utils.plot import (
    plot_u_space, plot_y_space,
    plot_pdf_normalization,
    # plot_pred_vs_true
)

from models.qkal import QKAL
from models.linear_model import LinearMomentsTorch
from models.small_deep_moments_mlp import SmallDeepMomentsMLP


def plot_histories(all_results):
    for deg, results in all_results.items():
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


def plot_best_val_bar(best_by_model: dict[str, tuple[int, float]]):
    names = list(best_by_model.keys())
    vals  = [best_by_model[n][1] for n in names]
    plt.figure(figsize=(6,4))
    plt.bar(names, vals)
    for i, v in enumerate(vals):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.ylabel("best val loss")
    plt.title("Best validation loss per model")
    plt.tight_layout(); plt.show()


def plot_val_vs_degree(all_results):
    per_model = {}
    for deg, res_by_model in all_results.items():
        for name, res in res_by_model.items():
            per_model.setdefault(name, {"deg": [], "val": []})
            per_model[name]["deg"].append(deg)
            per_model[name]["val"].append(res["best_val"])

    if any(len(v["deg"]) > 1 for v in per_model.values()):
        plt.figure(figsize=(7,5))
        for name, d in per_model.items():
            order = np.argsort(d["deg"])
            degs  = np.array(d["deg"])[order]
            vals  = np.array(d["val"])[order]
            plt.plot(degs, vals, marker="o", label=name)
        plt.xlabel("degree"); plt.ylabel("best val loss")
        plt.title("Validation loss vs degree")
        plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()



def plot_nll_hist_joint(per_model_nll_vectors: dict[str, np.ndarray], bins: int = 40):
    vecs = [v for v in per_model_nll_vectors.values() if v is not None and len(v)]
    if not vecs:
        return
    all_vals = np.concatenate(vecs, axis=0)
    plt.figure(figsize=(7,4))
    plt.hist(all_vals, bins=bins, alpha=0.9)
    p90 = np.percentile(all_vals, 90)
    plt.axvline(p90, linestyle="--", label=f"p90={p90:.3f}")
    plt.xlabel("NLL_i"); plt.ylabel("count"); plt.title("Per-sample NLL (all models)")
    plt.legend(); plt.tight_layout(); plt.show()


def plot_nll_box_per_model(per_model_nll_vectors: dict[str, np.ndarray]):
    names, data = [], []
    for name, vec in per_model_nll_vectors.items():
        if vec is not None and len(vec):
            names.append(name); data.append(vec)
    if not data:
        return
    plt.figure(figsize=(7,4))
    plt.boxplot(data, labels=names, showfliers=False)
    plt.ylabel("NLL_i")
    plt.title("Per-sample NLL per model")
    plt.tight_layout(); plt.show()



def evaluate_full(
    model,
    X_te: np.ndarray,
    y_te: np.ndarray,
    cfg: ReconstructionConfig,
    *,
    y_ref_raw: np.ndarray,
    qt,
    nll_space: str = "log1p",
    do_plots: bool = False,
    return_nll_vector: bool = True,
) -> tuple[dict, np.ndarray | None]:
    X_te = np.asarray(X_te, dtype=np.float32)
    y_te = np.asarray(y_te, dtype=np.float32)

    nll_per = None
    if return_nll_vector:
        try:
            nll, nll_per = eval_nll(
                model, X_te, y_te, cfg,
                space=nll_space, y_ref_raw=y_ref_raw, qt=qt,
                return_per_sample=True
            )
            nll = float(nll)
        except TypeError:
            nll = float(eval_nll(
                model, X_te, y_te, cfg,
                space=nll_space, y_ref_raw=y_ref_raw, qt=qt
            ))
    else:
        nll = float(eval_nll(
            model, X_te, y_te, cfg,
            space=nll_space, y_ref_raw=y_ref_raw, qt=qt
        ))

    yhat = predict_mean_from_density(
        model, X_te, y_te, cfg,
        space="raw", y_ref_raw=y_ref_raw, qt=qt
    )

    out = {
        "NLL": nll,
        "RMSE": rmse(y_te, yhat),
        "MAE": mae(y_te, yhat),
        "corr": float(np.corrcoef(y_te, yhat)[0, 1]),
        "mean_y": float(y_te.mean()),
        "mean_yhat": float(yhat.mean()),
    }

    return out, nll_per



def build_parser():
    ap = argparse.ArgumentParser("QKAL/MLP/LINEAR — housing (+ NLL plots; no E[Y|x] scatter)")
    ap.add_argument("--csv", default="h_prices.csv")
    ap.add_argument("--target", default="House_Price")
    ap.add_argument("--scale_x", action="store_true")
    ap.add_argument("--degrees", default="3", help="np. '3,4,5'")
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plots", action="store_true", help="rysuj wykresy (historia/val + NLL)")
    ap.add_argument("--nll_space", choices=["raw", "log1p"], default="log1p",
                    help="Przestrzeń dla NLL (domyślnie 'log1p')")
    ap.add_argument("--kde_rule", choices=["silverman", "scott", "iqr"], default="silverman",
                    help="Reguła bandwidth dla KDE")
    ap.add_argument("--kde_mult", type=float, default=1.5,
                    help="Mnożnik dla bandwidth KDE")
    ap.add_argument("--grid_size", type=int, default=256,
                    help="Rozmiar siatki u/S")
    return ap


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    np.random.seed(args.seed)

    # dane
    df = pd.read_csv(args.csv)
    fe = engineer_housing_features(df, target_col=args.target, scale=False, add_log_target=False)
    X_df = fe["X"]; y_sr = fe["y"]

    X = X_df.to_numpy(dtype=np.float32)
    y = y_sr.to_numpy(dtype=np.float32)

    if args.scale_x:
        X = StandardScaler().fit_transform(X).astype(np.float32)

    # split
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(y)); rng.shuffle(idx)
    n_test = int(args.test_ratio * len(y))
    te_idx = idx[:n_test]
    tv_idx = idx[n_test:]

    X_te, y_te = X[te_idx], y[te_idx]
    X_tv, y_tv = X[tv_idx], y[tv_idx]

    tr_dl, va_dl, qt = make_loaders(
        X_tv, y_tv, batch_size=args.batch_size, val_ratio=args.val_ratio, seed=args.seed
    )

    # trening
    degrees = [int(d.strip()) for d in args.degrees.split(",") if d.strip()]
    all_results: dict[int, dict[str, dict]] = {}
    best_by_model: dict[str, tuple[int, float]] = {}
    best_models: dict[str, tuple[object, ReconstructionConfig]] = {}

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
        cfg.calibration_mode = CalibrationMode.SOFTPLUS
        cfg.grid_size = args.grid_size
        cfg.kde_bw_rule = args.kde_rule
        cfg.kde_bw_mult = args.kde_mult
        cfg.nll_space = args.nll_space

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
                best_models[name] = (res["model"], cfg)

        all_results[deg] = results_deg

    print("\nBest degree per model (by val loss):")
    for name, (deg, val) in best_by_model.items():
        print(f"{name}: degree={deg}, val_loss={val:.6f}")

    if args.plots:
        plot_histories(all_results)
        plot_best_val_bar(best_by_model)
        plot_val_vs_degree(all_results)

    rows = []
    per_model_nll_vectors: dict[str, np.ndarray | None] = {}
    for name, (model, cfg_used) in best_models.items():
        print(f"\n>>> TEST EVAL: {name.lower()} (best degree={cfg_used.degree}, val={best_by_model[name][1]:.6f})")
        metrics, nll_vec = evaluate_full(
            model, X_te, y_te, cfg_used,
            y_ref_raw=y_tv,
            qt=qt,
            nll_space=args.nll_space,
            do_plots=False,
            return_nll_vector=True
        )
        rows.append({"model": name, **metrics})
        per_model_nll_vectors[name] = nll_vec

    if rows:
        df_sum = pd.DataFrame(rows)
        print("\n=== SUMMARY (holdout / test) ===")
        cols = ["model","NLL","RMSE","MAE","corr","mean_y","mean_yhat"]
        print(df_sum[cols].to_string(index=False))
        os.makedirs("runs", exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        df_sum.to_csv(f"runs/housing_summary_{ts}.csv", index=False)

    if args.plots:
        any_vec = any((v is not None and len(v)) for v in per_model_nll_vectors.values())
        if any_vec:
            plot_nll_hist_joint(per_model_nll_vectors, bins=40)
            plot_nll_box_per_model(per_model_nll_vectors)
        else:
            print("[info] eval_nll nie zwrócił per-sample NLL — wykresy NLL pominięte.")

    if args.plots and rows:
        best_idx = int(np.argmin([r["NLL"] for r in rows]))
        best_name = rows[best_idx]["model"]
        model, cfg_used = best_models[best_name]
        N = min(256, len(y_te))
        y_grid_t, f_y_batch_t, u_grid, rho_u_batch, _, p_marg_y_t = density_from_model(
            model, X_te[:N], y_tv, cfg_used, qt=qt, space="raw"
        )
        plot_u_space(u_grid, rho_u_batch, n_show=min(8, N))
        plot_y_space(y_grid_t, f_y_batch_t, y_data=y_te[:N], p_marg_y=p_marg_y_t, n_show=min(8, N))


def run(**kwargs):
    def to_argv(k: str, v):
        flag = f"--{k.replace('_','-')}"
        if isinstance(v, bool):
            return [flag] if v else []
        else:
            return [flag, str(v)]
    argv = []
    for k, v in kwargs.items():
        argv += to_argv(k, v)
    return main(argv)


if __name__ == "__main__":
     preset = [
    #     "--csv", "h_prices.csv",
    #     "--degrees", "3",
    #     "--epochs", "100",
    #     "--batch_size", "2048",
    #     "--nll_space", "log1p",
    #     "--kde_rule", "iqr",
    #     "--kde_mult", "1.5",
    #     "--grid_size", "512",
         "--plots",
     ]
     main(preset)
    #main()