"""
Feature Engineering module for Mumbai Urban Heat Island Modeling.
Provides formulas for remote sensing indices, coastal distance transforms,
spatial normalization, and coordinate scaling for PINN derivatives.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt


def compute_spectral_indices(b2, b3, b4, b8, b11, b12):
    """
    Computes spectral indices from Sentinel-2 surface reflectance bands.
    b2: Blue, b3: Green, b4: Red, b8: NIR, b11: SWIR-1, b12: SWIR-2
    """
    eps = 1e-7

    # NDVI: (NIR - Red) / (NIR + Red)
    ndvi = (b8 - b4) / (b8 + b4 + eps)

    # NDBI: (SWIR1 - NIR) / (SWIR1 + NIR)
    ndbi = (b11 - b8) / (b11 + b8 + eps)

    # NDWI: (Green - NIR) / (Green + NIR)
    ndwi = (b3 - b8) / (b3 + b8 + eps)

    # MNDWI: (Green - SWIR1) / (Green + SWIR1)
    mndwi = (b3 - b11) / (b3 + b11 + eps)

    return {
        "ndvi": np.clip(ndvi, -1.0, 1.0),
        "ndbi": np.clip(ndbi, -1.0, 1.0),
        "ndwi": np.clip(ndwi, -1.0, 1.0),
        "mndwi": np.clip(mndwi, -1.0, 1.0)
    }


def compute_distance_to_sea(is_water_grid, pixel_size_m=30.0):
    """
    Computes Euclidean distance to open water / coast using exact distance transform.
    Returns distance in kilometers.
    """
    # 0 where water, >0 on land
    dist_pixels = distance_transform_edt(~is_water_grid)
    dist_km = (dist_pixels * pixel_size_m) / 1000.0
    return dist_km


def scale_coordinates(lon, lat, bounds=None):
    """
    Normalizes coordinates to [-1, 1] range.
    Crucial for PINN numerical stability and non-dimensional spatial derivatives.
    """
    if bounds is None:
        min_lon, max_lon = np.nanmin(lon), np.nanmax(lon)
        min_lat, max_lat = np.nanmin(lat), np.nanmax(lat)
    else:
        min_lon, max_lon = bounds["min_lon"], bounds["max_lon"]
        min_lat, max_lat = bounds["min_lat"], bounds["max_lat"]

    x_norm = 2.0 * (lon - min_lon) / (max_lon - min_lon + 1e-8) - 1.0
    y_norm = 2.0 * (lat - min_lat) / (max_lat - min_lat + 1e-8) - 1.0
    return x_norm, y_norm
