import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler

def engineer_housing_features(
    df: pd.DataFrame,
    *,
    target_col: str = "House_Price",
    scale: bool = False,
    scaler: StandardScaler | None = None,
    current_year: int | None = None,
    add_log_target: bool = False
):
    df = df.copy()

    required = [
        "Square_Footage","Num_Bedrooms","Num_Bathrooms","Year_Built",
        "Lot_Size","Garage_Size","Neighborhood_Quality", target_col
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    eps = 1e-9
    if current_year is None:
        current_year = datetime.now().year

    df["Year_Built"] = pd.to_numeric(df["Year_Built"], errors="coerce").fillna(0).astype(int)

    den_bed  = df["Num_Bedrooms"].to_numpy(dtype=float)
    den_bath = df["Num_Bathrooms"].to_numpy(dtype=float)
    den_bed  = np.where(den_bed  <= 0, eps, den_bed)
    den_bath = np.where(den_bath <= 0, eps, den_bath)

    df["House_Age"]              = current_year - df["Year_Built"]
    df["Sqft_per_Bedroom"]       = df["Square_Footage"] / den_bed
    df["Sqft_per_Bathroom"]      = df["Square_Footage"] / den_bath
    df["Lot_per_Sqft"]           = df["Lot_Size"] / (df["Square_Footage"].to_numpy(dtype=float) + eps)
    df["Baths_per_Bedroom"]      = den_bath / den_bed
    df["Beds_plus_Baths"]        = df["Num_Bedrooms"] + df["Num_Bathrooms"]
    df["Garage_Flag"]            = (df["Garage_Size"] > 0).astype(int)
    df["Garage_per_Bedroom"]     = df["Garage_Size"] / den_bed

    df["Bedrooms_x_Bathrooms"]   = df["Num_Bedrooms"] * df["Num_Bathrooms"]
    df["Quality_Sq"]             = df["Neighborhood_Quality"] ** 2
    df["Age_Sq"]                 = df["House_Age"] ** 2

    if add_log_target:
        df["log1p_" + target_col] = np.log1p(df[target_col])

    feature_cols = [
        # oryginalne
        "Square_Footage","Num_Bedrooms","Num_Bathrooms","Year_Built",
        "Lot_Size","Garage_Size","Neighborhood_Quality",
        # engineered
        "House_Age","Sqft_per_Bedroom","Sqft_per_Bathroom","Lot_per_Sqft",
        "Baths_per_Bedroom","Beds_plus_Baths","Garage_Flag","Garage_per_Bedroom",
        "Bedrooms_x_Bathrooms","Quality_Sq","Age_Sq"
    ]

    X = df[feature_cols].astype(float)
    y = df[target_col].astype(float)

    fitted_scaler = scaler
    if scale:
        if fitted_scaler is None:
            fitted_scaler = StandardScaler()
            X_scaled = fitted_scaler.fit_transform(X)
        else:
            X_scaled = fitted_scaler.transform(X)
        X = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)

    out = {"X": X, "y": y, "scaler": fitted_scaler, "feature_names": feature_cols}
    if add_log_target:
        out["y_log1p"] = df["log1p_" + target_col]
    return out
