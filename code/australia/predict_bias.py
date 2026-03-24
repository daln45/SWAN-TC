# -*- coding: utf-8 -*-
"""
Predict SWAN-C bias for Australian coastal sites using the UK-trained
CatBoost multi-target model.

This is Step 6 of the Australia transfer-learning pipeline.

Usage
-----
    python predict_bias.py

Inputs
------
    data/aus/swan_depth_data_2021_01.csv
    data/aus/wind_wave_initial_data_2021_01_aus.csv
    weights/best_multi_bias_model.cbm
    weights/scaler_multi_bias.pkl

Output
------
    results/aus_bias_predictions/results_2021_01_Aus_Pred.xlsx
        Columns: Buoy_ID, Time, Pred_Bias_Hs, Pred_Bias_Tm
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR    = Path(__file__).parent.parent.parent   # repo root
WEIGHTS_DIR = BASE_DIR / "weights"
DATA_DIR    = BASE_DIR / "data" / "aus"
RESULT_DIR  = BASE_DIR / "results" / "aus_bias_predictions"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

RESULT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(RESULT_DIR / "predict_log.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config:
    # Model weights
    model_path  = WEIGHTS_DIR / "best_multi_bias_model.cbm"
    scaler_path = WEIGHTS_DIR / "scaler_multi_bias.pkl"

    # Input data
    depth_file    = DATA_DIR / "swan_depth_data_2021_01.csv"
    wind_wave_file = DATA_DIR / "wind_wave_initial_data_2021_01_aus.csv"

    # Output
    output_file = RESULT_DIR / "results_2021_01_Aus_Pred.xlsx"

    # Grid
    grid_spacing = 50
    depth_index  = 200


# ---------------------------------------------------------------------------
# Supporting classes
# ---------------------------------------------------------------------------

class SpatialCoordManager:
    """Load per-buoy depth time-series from a CSV file."""

    @classmethod
    def load_coordinates(cls, depth_df: pd.DataFrame) -> dict:
        if not pd.api.types.is_datetime64_any_dtype(depth_df["time"]):
            depth_df["time"] = pd.to_datetime(
                depth_df["time"].astype(str).str.replace(r"\.0$", "", regex=True),
                format="%Y%m%d%H",
                errors="coerce",
            )
        depth_df = depth_df.dropna(subset=["time"])
        depth_df["time"] = depth_df["time"].dt.floor("h")

        depth_map: dict = {}
        for buoy_id, grp in depth_df.groupby("id"):
            depth_map[buoy_id] = {}
            if "depth200" in grp.columns:
                for t, d in grp.set_index("time")["depth200"].items():
                    depth_map[buoy_id][t] = np.array([d], dtype=np.float32)

        x = Config.grid_spacing * np.array([Config.depth_index], dtype=np.float32)
        y = np.zeros(1, dtype=np.float32)
        return {"x": x, "y": y, "depth_map": depth_map}


class SimpleFeatureProcessor:
    """Add trigonometric direction encodings required by the UK model."""

    @staticmethod
    def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Fill optional columns
        for col in ("alpc",):
            if col not in df.columns:
                df[col] = 0.0

        # Coerce to float32
        for col in ("swh", "wind_speed", "wind_direction", "mwd", "mwp", "alpc", "tide"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)

        # Sine / cosine encoding — must match UK training order
        if "wind_direction" in df.columns:
            wd_rad = np.deg2rad(df["wind_direction"])
            df["sin_wd"] = np.sin(wd_rad).astype(np.float32)
            df["cos_wd"] = np.cos(wd_rad).astype(np.float32)

        if "mwd" in df.columns:
            mwd_rad = np.deg2rad(df["mwd"])
            df["sin_mwd"] = np.sin(mwd_rad).astype(np.float32)
            df["cos_mwd"] = np.cos(mwd_rad).astype(np.float32)

        return df


class HierarchicalScaler:
    """Placeholder for deserialising the pickled UK scaler."""

    def __init__(self) -> None:
        self.bias_hs_scaler   = StandardScaler()
        self.bias_tm_scaler   = StandardScaler()
        self.depth_scaler     = StandardScaler()
        self.feature_scalers: dict = {}

    # Stubs required for pickle compatibility
    def fit(self, df: pd.DataFrame) -> None: ...
    def transform(self, df: pd.DataFrame) -> None: ...

    def inverse_transform_targets(
        self,
        preds_hs: np.ndarray,
        preds_tm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        hs = self.bias_hs_scaler.inverse_transform(preds_hs.reshape(-1, 1)).ravel()
        tm = self.bias_tm_scaler.inverse_transform(preds_tm.reshape(-1, 1)).ravel()
        return hs, tm


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class AusPredictDataset:
    """Load and merge depth + wind/wave CSV files for Australian buoys."""

    def __init__(self) -> None:
        self.coord_system = self._load_spatial(Config.depth_file)
        self.df = self._load_and_merge()
        if not self.df.empty:
            self.df = SimpleFeatureProcessor.add_basic_features(self.df)
            logger.info("Dataset loaded: %d samples", len(self.df))
        else:
            logger.error("Merged dataframe is empty — check input file paths.")

    def _load_spatial(self, depth_path: Path) -> dict:
        if not depth_path.exists():
            logger.warning("Depth file not found: %s", depth_path)
            return {"depth_map": {}}
        depth_df = pd.read_csv(depth_path, dtype={"time": str})
        depth_df["time"] = (
            depth_df["time"].astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .apply(lambda x: str(int(float(x))) if "E" in x.upper() else x)
        )
        return SpatialCoordManager.load_coordinates(depth_df)

    def _load_and_merge(self) -> pd.DataFrame:
        if not Config.depth_file.exists() or not Config.wind_wave_file.exists():
            logger.error("One or more input files are missing.")
            return pd.DataFrame()

        depth    = pd.read_csv(Config.depth_file,     dtype={"time": str})
        wind_wave = pd.read_csv(Config.wind_wave_file, dtype={"time": str})

        rename = {
            "Time": "time", "ID": "id",
            "Wind_Speed": "wind_speed", "Wind_Direction": "wind_direction",
            "SWH": "swh", "MWD": "mwd", "MWP": "mwp", "Tide": "tide",
            "wind_Speed": "wind_speed",
        }
        wind_wave = wind_wave.rename(columns=rename)
        wind_wave.columns = [c.lower() if c in ("TIME", "ID") else c for c in wind_wave.columns]

        for df in (depth, wind_wave):
            df["time"] = (
                df["time"].astype(str)
                .str.replace(r"\.0$", "", regex=True)
                .apply(lambda x: str(int(float(x))) if "E" in x.upper() else x)
            )
            df["time"] = pd.to_datetime(df["time"], format="%Y%m%d%H", errors="coerce").dt.floor("h")

        merged = depth.merge(wind_wave, on=["id", "time"], how="inner")
        logger.info("Merged dataset: %d rows", len(merged))
        return merged


# ---------------------------------------------------------------------------
# Main prediction pipeline
# ---------------------------------------------------------------------------

def _apply_scaler(df: pd.DataFrame, scaler: HierarchicalScaler) -> pd.DataFrame:
    """Normalise physical and depth features using the UK scaler."""
    feat_map = {
        "swh": "swh",         "wind_speed": "wind_speed",
        "mwp": "mwp",         "tide": "tide",
        "sin_wd": "sin_wd",   "cos_wd": "cos_wd",
        "sin_mwd": "sin_mwd", "cos_mwd": "cos_mwd",
    }
    for key, col in feat_map.items():
        if key in scaler.feature_scalers and col in df.columns:
            sc   = scaler.feature_scalers[key]
            vals = df[col].values.reshape(-1, 1)
            if hasattr(sc, "feature_names_in_"):
                vals = pd.DataFrame(vals, columns=sc.feature_names_in_)
            df[col] = sc.transform(vals).ravel()

    if hasattr(scaler, "depth_scaler") and "depth200" in df.columns:
        sc   = scaler.depth_scaler
        vals = df["depth200"].values.reshape(-1, 1)
        if hasattr(sc, "feature_names_in_"):
            vals = pd.DataFrame(vals, columns=sc.feature_names_in_)
        df["depth200"] = sc.transform(vals).ravel()

    return df


def main() -> None:
    logger.info("=" * 60)
    logger.info("  AUS BIAS PREDICTION  (UK CatBoost multi-target model)")
    logger.info("=" * 60)

    # --- Load model & scaler ---
    if not Config.model_path.exists():
        logger.error("Model file not found: %s", Config.model_path)
        return
    model: CatBoostRegressor = CatBoostRegressor()
    model.load_model(str(Config.model_path))

    with open(Config.scaler_path, "rb") as fh:
        scaler: HierarchicalScaler = pickle.load(fh)

    # --- Prepare data ---
    dataset = AusPredictDataset()
    if dataset.df.empty:
        return

    df = _apply_scaler(dataset.df.copy(), scaler)

    # Feature order MUST match UK training  (9 columns)
    FEATURE_ORDER = [
        "swh", "wind_speed", "mwp", "tide",
        "sin_wd", "cos_wd", "sin_mwd", "cos_mwd",
        "depth200",
    ]
    X = np.column_stack([df[c].values for c in FEATURE_ORDER]).astype(np.float32)

    # --- Predict ---
    logger.info("Running CatBoost inference  (%d samples) …", len(X))
    preds_scaled = model.predict(X)

    pred_hs, pred_tm = scaler.inverse_transform_targets(
        preds_scaled[:, 0], preds_scaled[:, 1]
    )

    # --- Save ---
    result_df = pd.DataFrame({
        "Buoy_ID":      dataset.df["id"].values,
        "Time":         dataset.df["time"].dt.strftime("%Y%m%d%H").values,
        "Pred_Bias_Hs": pred_hs,
        "Pred_Bias_Tm": pred_tm,
    })
    result_df.to_excel(Config.output_file, index=False)
    logger.info("Saved → %s", Config.output_file)
    logger.info(
        "Stats:  Hs bias mean=%.4f  Tm bias mean=%.4f",
        pred_hs.mean(), pred_tm.mean(),
    )


if __name__ == "__main__":
    main()
