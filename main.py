from qkal.config import QKALReconstructionConfig
from qkal.train import train_qkal_from_arrays
from qkal.density import density_from_qkal
from qkal.plot import plot_y_space, plot_u_space
from qkal.eval import eval_nll, predict_mean_from_density, rmse, mae, baseline_lr_metrics
import re
import numpy as np
import pandas as pd

CSV_PATH = r"realtor-data.zip.csv"

def _find_price_column(df: pd.DataFrame) -> str:
    """price / saleprice / sale_price (case-insensitive)."""
    candidates = [c for c in df.columns if re.fullmatch(r"(sale[_ ]?price|price)", c, flags=re.I)]
    if not candidates:
        raise KeyError("Nie znaleziono kolumny 'price' / 'sale_price' / 'SalePrice' w CSV.")
    return candidates[0]

def _to_numeric_price(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.number):
        return s.astype(float)
    s = s.astype(str).str.replace(r"[^\d\.]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def load_your_data(csv_path: str = CSV_PATH, numeric_only: bool = True, log_price: bool = False):
    df = pd.read_csv(csv_path, low_memory=False)

    price_col = _find_price_column(df)
    df[price_col] = _to_numeric_price(df[price_col])

    # --- CLEANING ---

    df = df.dropna(subset=[price_col])
    df = df[df[price_col] > 0]


    df = df[df[price_col] != 2147483600]


    df = df[df[price_col].between(10_000, 20_000_000)]


    p99_9 = df[price_col].quantile(0.999)
    df.loc[df[price_col] > p99_9, price_col] = p99_9
    # --- END CLEANING ---

    y = df[price_col].to_numpy(dtype=float)


    if numeric_only:
        X_df = df.drop(columns=[price_col]).select_dtypes(include=["number"]).copy()
        X_df = X_df.fillna(X_df.median(numeric_only=True))
    else:
        X_df = df.drop(columns=[price_col]).copy()
        cat_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
        X_df = pd.get_dummies(X_df, columns=cat_cols, dummy_na=True)
        X_df = X_df.fillna(X_df.median(numeric_only=True))

    X = X_df.to_numpy(dtype=float)
    return X, y


if __name__ == "__main__":
    config = QKALReconstructionConfig()
    #zmienione podstawowe ustawienia, aby sprawdzić działąnie eval i całęgo programu, można wrócić do podstawowych z config na końcu
    config.degree = 3
    config.hidden_dim = 64
    config.batch_size = 4096 #sprawdzic batch size ogólnie w kodzie, czy działa poprawnie
    config.grid_size = 512

    X, y = load_your_data(log_price=False)

    if getattr(config, "input_dim", None) != X.shape[1]:
        config.input_dim = X.shape[1]

    model, qt = train_qkal_from_arrays(X, y, config)

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
