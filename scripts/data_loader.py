"""
Data loader and preprocessor for Mumbai UHI modeling.
Implements:
- Data ingestion and land-pixel filtering.
- Continuous spatial feature standardization.
- Contiguous spatial holdout splitting (to test true spatial generalization).
- PyTorch Dataset wrappers for GPU-accelerated PINN training.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from configs.config import (
    PROCESSED_DATA_DIR, FEATURE_COLUMNS, TARGET_COLUMN,
    MUMBAI_BOUNDS
)
from scripts.features import scale_coordinates


class MumbaiThermalDataset(Dataset):
    """
    PyTorch Dataset yielding coordinates (x, y, t) and environmental features.
    """
    def __init__(self, coords, features, targets):
        self.coords = torch.tensor(coords, dtype=torch.float32)
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.coords[idx], self.features[idx], self.targets[idx]


def load_mumbai_data(holdout_zone=4, val_fraction=0.15, random_seed=42):
    """
    Loads Mumbai features parquet file, applies land filtering, normalizes features,
    and returns spatial train, random validation, and spatial holdout sets.

    holdout_zone: contiguous spatial zone held out for unseen generalization testing.
                  Zone 4 = Eastern Suburbs South (Govandi, Chembur, Kurla).
    """
    parquet_path = PROCESSED_DATA_DIR / "mumbai_features.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {parquet_path}. Run generate_sample_mumbai_dataset.py first.")

    df = pd.read_parquet(parquet_path)
    # Filter to land pixels for thermal modeling
    df_land = df[df["is_land"] == 1].copy().reset_index(drop=True)

    # 1. Coordinate Normalization to [-1, 1]
    x_norm, y_norm = scale_coordinates(df_land["lon"].values, df_land["lat"].values, MUMBAI_BOUNDS)
    df_land["x_norm"] = x_norm
    df_land["y_norm"] = y_norm
    # Time variable t (normalized: e.g., 0.5 for afternoon Landsat overpass ~11:00 AM)
    df_land["t_norm"] = 0.5

    # 2. Contiguous Spatial Holdout Split
    # Hold out zone `holdout_zone` completely
    is_holdout = df_land["spatial_zone"] == holdout_zone
    df_train_val = df_land[~is_holdout].copy()
    df_spatial_test = df_land[is_holdout].copy()

    # Split train_val into train and validation sets
    np.random.seed(random_seed)
    n_train_val = len(df_train_val)
    val_indices = np.random.choice(n_train_val, size=int(n_train_val * val_fraction), replace=False)
    train_mask = np.ones(n_train_val, dtype=bool)
    train_mask[val_indices] = False

    df_train = df_train_val.iloc[train_mask].copy()
    df_val = df_train_val.iloc[~train_mask].copy()

    # 3. Feature Scaling (Fit on Train Only)
    scaler = StandardScaler()
    X_train_feat = scaler.fit_transform(df_train[FEATURE_COLUMNS].values)
    X_val_feat = scaler.transform(df_val[FEATURE_COLUMNS].values)
    X_test_feat = scaler.transform(df_spatial_test[FEATURE_COLUMNS].values)

    # Coordinates: [x_norm, y_norm, t_norm]
    coords_train = df_train[["x_norm", "y_norm", "t_norm"]].values
    coords_val = df_val[["x_norm", "y_norm", "t_norm"]].values
    coords_test = df_spatial_test[["x_norm", "y_norm", "t_norm"]].values

    y_train = df_train[TARGET_COLUMN].values
    y_val = df_val[TARGET_COLUMN].values
    y_test = df_spatial_test[TARGET_COLUMN].values

    data_dict = {
        "df_all": df_land,
        "df_train": df_train,
        "df_val": df_val,
        "df_test": df_spatial_test,
        "scaler": scaler,
        "feature_names": FEATURE_COLUMNS,
        "X_train": X_train_feat,
        "y_train": y_train,
        "coords_train": coords_train,
        "X_val": X_val_feat,
        "y_val": y_val,
        "coords_val": coords_val,
        "X_test": X_test_feat,
        "y_test": y_test,
        "coords_test": coords_test,
    }

    print(f"Loaded Mumbai Dataset: {len(df_land):,} land pixels")
    print(f"Train set: {len(df_train):,} pixels")
    print(f"Validation set: {len(df_val):,} pixels")
    print(f"Contiguous Spatial Holdout Test set (Zone {holdout_zone}): {len(df_spatial_test):,} pixels")

    return data_dict
