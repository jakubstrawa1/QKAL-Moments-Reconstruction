# utils/data.py
import re
import numpy as np
import pandas as pd
from typing import Tuple

CSV_PRICE_PAT = r"(sale[_ ]?price|price)"

def _find_price_column(df: pd.DataFrame) -> str:
    cand = [c for c in df.columns if re.fullmatch(CSV_PRICE_PAT, c, flags=re.I)]
    if not cand:
        raise KeyError("Nie znaleziono kolumny 'price' / 'sale_price' / 'SalePrice' w CSV.")
    return cand[0]

def _to_numeric_price(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.number):
        return s.astype(float)
    s = s.astype(str).str.replace(r"[^\d\.]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def load_realtor_csv(
    csv_path: str,
    numeric_only: bool = False,
    log_price: bool = False,
    low_card_max: int = 50,
    drop_text: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path, low_memory=False)

    # --- price + cleaning ---
    price_col = _find_price_column(df)
    df[price_col] = _to_numeric_price(df[price_col])
    df = df.dropna(subset=[price_col])
    df = df[df[price_col] > 0]
    df = df[df[price_col] != 2_147_483_600]
    df = df[df[price_col].between(10_000, 20_000_000)]
    p999 = df[price_col].quantile(0.999)
    df.loc[df[price_col] > p999, price_col] = p999

    y = df[price_col].to_numpy(dtype=np.float32)
    if log_price:
        y = np.log1p(y).astype(np.float32)

    X_all = df.drop(columns=[price_col]).copy()
    if numeric_only:
        X_df = X_all.select_dtypes(include=["number"]).copy()
        X_df = X_df.fillna(X_df.median(numeric_only=True))
        return X_df.to_numpy(np.float32), y

    if drop_text:
        drop_like = ["address","street","st.","unit","apt","suite","desc","description","title",
                     "link","url","mls","listing","id","phone","email","agent","broker"]
        X_all = X_all.drop(columns=[c for c in X_all.columns if any(t in c.lower() for t in drop_like)],
                           errors="ignore")

    num_df = X_all.select_dtypes(include=["number"]).copy()
    num_df = num_df.fillna(num_df.median(numeric_only=True))

    cat_df = X_all.select_dtypes(include=["object","category","bool"]).copy()
    nun = cat_df.nunique(dropna=False)
    low_cols  = nun[nun <= low_card_max].index.tolist()
    high_cols = nun[nun >  low_card_max].index.tolist()

    ohe = (pd.get_dummies(cat_df[low_cols], dummy_na=True).fillna(0)) if low_cols else pd.DataFrame(index=cat_df.index)
    freq_df = pd.DataFrame(index=cat_df.index)
    for c in high_cols:
        vc = cat_df[c].astype(str).value_counts(dropna=False)
        freq_df[c + "_freq"] = cat_df[c].astype(str).map(vc).astype("float32")
    freq_df = freq_df.fillna(0)

    X_df = pd.concat([num_df, ohe, freq_df], axis=1).replace([np.inf, -np.inf], 0)
    return X_df.to_numpy(np.float32), y
