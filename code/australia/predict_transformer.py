# -*- coding: utf-8 -*-
"""
SWAN-T prediction for Australian coastal sites with additive bias correction.

This is Step 7 of the Australia transfer-learning pipeline.

Pipeline
--------
    1. Load UK-trained Transformer weights.
    2. Build a per-timestep dataset from depth + ERA5 wind/wave CSV files.
    3. Run inference to obtain raw Hs / Tm spatial profiles.
    4. Apply additive bias correction: final_Hs = raw_Hs − Pred_Bias_Hs.
    5. Enforce breaking limit: Hs ≤ 0.78 × depth.
    6. Save results as a multi-sheet Excel workbook.

Usage
-----
    python predict_transformer.py

Inputs
------
    data/aus/swan_depth_data_2021_01.csv
    data/aus/wind_wave_initial_data_2021_01_aus.csv
    results/aus_bias_predictions/results_2021_01_Aus_Pred.xlsx
    weights/best_model_exp_5_100_percent.pth
    weights/scaler_exp_5_100_percent.pkl

Output
------
    results/aus_transformer/Aus_Final_Pred_2021_Additive_Limit.xlsx
        Sheets: Meta | Hs_Corrected | Tm_Corrected | Hs_Raw | Tm_Raw
"""

from __future__ import annotations

import gc
import logging
import math
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR    = Path(__file__).parent.parent.parent   # repo root
WEIGHTS_DIR = BASE_DIR / "weights"
DATA_DIR    = BASE_DIR / "data" / "aus"
BIAS_FILE   = BASE_DIR / "results" / "aus_bias_predictions" / "results_2021_01_Aus_Pred.xlsx"
RESULT_DIR  = BASE_DIR / "results" / "aus_transformer"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class PredictionConfig:
    # ----- Transformer architecture (must match UK training) -----
    d_model          = 256
    nhead            = 8
    num_layers       = 6
    dim_feedforward  = 512
    dropout          = 0.1
    activation       = "gelu"
    input_channels   = {"spatial": 1, "physical": 7, "output": 2}

    # ----- Spatial grid -----
    coord_cols    = ["depth"]
    depth_indices = list(range(10, 200, 3))[:64]   # 64 active nodes
    spatial_points = 64
    grid_spacing  = 50

    # ----- Data selection -----
    predict_years = [2021]
    test_months   = [1]

    # ----- Inference -----
    batch_size = 64
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Paths -----
    model_path  = WEIGHTS_DIR / "best_model_exp_5_100_percent.pth"
    scaler_path = WEIGHTS_DIR / "scaler_exp_5_100_percent.pkl"
    data_files  = {
        "depth": DATA_DIR / "swan_depth_data_2021_01.csv",
        "wind":  DATA_DIR / "wind_wave_initial_data_2021_01_aus.csv",
    }
    bias_file   = BIAS_FILE
    result_dir  = RESULT_DIR


# ---------------------------------------------------------------------------
# Supporting classes
# ---------------------------------------------------------------------------

class SpatialCoordManager:
    """Parse depth columns (depth010 … depth199) into per-buoy time-maps."""

    @classmethod
    def load_coordinates(
        cls,
        depth_df: pd.DataFrame,
        months_filter: list[int] | None = None,
    ) -> dict:
        depth_map: dict = {}
        pattern  = re.compile(r"^(depth)(\d{3})$")

        if "id" not in depth_df.columns or "time" not in depth_df.columns:
            return {"x": [], "y": [], "depth_map": {}}

        depth_df["time"] = (
            depth_df["time"].astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .apply(lambda x: str(int(float(x))) if "E" in x.upper() else x)
        )
        depth_df["time"] = pd.to_datetime(
            depth_df["time"], format="%Y%m%d%H", errors="coerce"
        ).dt.floor("h")
        depth_df = depth_df.dropna(subset=["time"])

        if months_filter:
            depth_df = depth_df[depth_df["time"].dt.month.isin(months_filter)]

        coord_map: dict = {}
        for col in depth_df.columns:
            if col in ("id", "time"):
                continue
            m = pattern.match(col)
            if m:
                var_type, idx_str = m.groups()
                idx_num = int(idx_str)
                if var_type in PredictionConfig.coord_cols and idx_num in PredictionConfig.depth_indices:
                    coord_map[idx_str] = {var_type: col}

        for buoy_id in depth_df["id"].unique():
            buoy_df = depth_df[depth_df["id"] == buoy_id]
            depth_map[buoy_id] = {}
            for _, row in buoy_df.iterrows():
                vals = [
                    row[coord_map[f"{i:03d}"]["depth"]]
                    if f"{i:03d}" in coord_map else np.nan
                    for i in PredictionConfig.depth_indices
                ]
                depth_map[buoy_id][row["time"]] = np.array(vals, dtype=np.float32)

        x = PredictionConfig.grid_spacing * np.array(
            PredictionConfig.depth_indices, dtype=np.float32
        )
        y = np.zeros(PredictionConfig.spatial_points, dtype=np.float32)
        return {"x": x, "y": y, "depth_map": depth_map}


class SimpleFeatureProcessor:
    """Normalise column names and add trig features."""

    @staticmethod
    def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        rename = {
            "Wind_Speed": "wind_speed", "Wind_Direction": "wind_direction",
            "SWH": "swh", "MWD": "mwd", "MWP": "mwp", "Tide": "tide",
            "wind_Speed": "wind_speed",
        }
        df = df.rename(columns=rename)
        df.columns = [c.lower() if c in ("TIME", "ID") else c for c in df.columns]

        for col in ("alpc", "tide"):
            if col not in df.columns:
                df[col] = 0.0

        for col in ("swh", "wind_speed", "wind_direction", "mwd", "mwp", "alpc", "tide"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
        return df


class ScenarioScaler:
    """Placeholder for deserialising the UK scaler pickle."""

    def __init__(self) -> None:
        self.hs_scaler       = StandardScaler()
        self.tm_scaler       = StandardScaler()
        self.depth_scaler    = StandardScaler()
        self.scenario_scalers: dict = {}
        self.is_fitted = False

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        depth_cols = [f"depth{i:03d}" for i in PredictionConfig.depth_indices]
        valid_depth = [c for c in depth_cols if c in df.columns]
        if valid_depth and hasattr(self.depth_scaler, "mean_"):
            df[valid_depth] = self.depth_scaler.transform(
                df[valid_depth].fillna(0).astype(np.float32)
            )
        for feat, sc in self.scenario_scalers.items():
            if feat in df.columns:
                df[feat] = sc.transform(df[[feat]].fillna(0).astype(np.float32))
        return df

    def inverse_transform_targets(self, targets: np.ndarray, var_type: str) -> np.ndarray:
        if var_type == "hs" and hasattr(self.hs_scaler, "mean_"):
            return self.hs_scaler.inverse_transform(targets)
        if var_type == "tm" and hasattr(self.tm_scaler, "mean_"):
            return self.tm_scaler.inverse_transform(targets)
        return targets


# ---------------------------------------------------------------------------
# Transformer model
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5_000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=PredictionConfig.dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class FeatureEmbedding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model             = d_model
        self.spatial_projection  = nn.Linear(PredictionConfig.input_channels["spatial"],  d_model)
        self.physical_projection = nn.Linear(PredictionConfig.input_channels["physical"], d_model)
        self.layer_norm          = nn.LayerNorm(d_model)
        self.dropout             = nn.Dropout(PredictionConfig.dropout)
        self.modal_weights       = nn.Parameter(torch.ones(2))

    def forward(
        self,
        spatial: torch.Tensor,
        physical: torch.Tensor,
        spatial_mask: torch.Tensor,
        physical_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_emb  = self.spatial_projection(spatial.transpose(1, 2))
        physical_emb = self.physical_projection(physical.transpose(1, 2))
        w = torch.softmax(self.modal_weights, dim=0)
        combined = self.dropout(self.layer_norm(w[0] * spatial_emb + w[1] * physical_emb))

        spatial_valid = spatial_mask.squeeze(1) if spatial_mask.dim() == 3 else spatial_mask
        if physical_mask.dim() == 2:
            physical_valid = physical_mask.all(dim=1, keepdim=True).expand(
                -1, PredictionConfig.spatial_points
            )
        else:
            physical_valid = physical_mask
        return combined, spatial_valid & physical_valid


class TransformerWavePredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_embedding = FeatureEmbedding(PredictionConfig.d_model)
        self.pos_encoder       = PositionalEncoding(
            PredictionConfig.d_model, max_len=PredictionConfig.spatial_points
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = PredictionConfig.d_model,
            nhead           = PredictionConfig.nhead,
            dim_feedforward = PredictionConfig.dim_feedforward,
            dropout         = PredictionConfig.dropout,
            activation      = PredictionConfig.activation,
            batch_first     = True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=PredictionConfig.num_layers
        )
        self.output_projection = nn.Sequential(
            nn.Linear(PredictionConfig.d_model,      PredictionConfig.d_model // 2), nn.GELU(),
            nn.Dropout(PredictionConfig.dropout),
            nn.Linear(PredictionConfig.d_model // 2, PredictionConfig.d_model // 4), nn.GELU(),
            nn.Dropout(PredictionConfig.dropout),
            nn.Linear(PredictionConfig.d_model // 4, PredictionConfig.input_channels["output"]),
        )

    def forward(
        self,
        spatial: torch.Tensor,
        physical: torch.Tensor,
        spatial_mask: torch.Tensor,
        physical_mask: torch.Tensor,
    ) -> torch.Tensor:
        emb, mask = self.feature_embedding(spatial, physical, spatial_mask, physical_mask)
        emb       = self.pos_encoder(emb.transpose(0, 1)).transpose(0, 1)
        out       = self.transformer_encoder(emb, src_key_padding_mask=~mask)
        return self.output_projection(out).permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PredictionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        coord_system: dict,
        scaler: ScenarioScaler,
    ) -> None:
        self.coord_system = coord_system
        _df = SimpleFeatureProcessor.add_basic_features(df)
        # Ensure dummy target columns exist for scaler
        for c in [f"hs{i:03d}" for i in PredictionConfig.depth_indices]:
            if c not in _df.columns:
                _df[c] = 0.0
        self.df = scaler.transform(_df).astype(
            {c: np.float32 for c in _df.select_dtypes("float64").columns}
        )
        del _df; gc.collect()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row     = self.df.iloc[idx]
        buoy_id = row["id"]
        ts      = row["time"]

        dm = self.coord_system["depth_map"]
        if buoy_id in dm and ts in dm[buoy_id]:
            d = dm[buoy_id][ts].astype(np.float32)
            n = PredictionConfig.spatial_points
            if len(d) < n:
                depth = np.zeros(n, dtype=np.float32); depth[:len(d)] = d
            else:
                depth = d[:n]
        else:
            depth = np.zeros(PredictionConfig.spatial_points, dtype=np.float32)

        spatial_input  = depth.reshape(1, -1)
        feats = np.array([
            row["swh"], row["wind_speed"], row["wind_direction"],
            row["mwd"],  row["mwp"],       row["alpc"], row["tide"],
        ], dtype=np.float32)
        physical_input = np.tile(feats, (PredictionConfig.spatial_points, 1)).T
        spatial_mask   = ~np.isnan(spatial_input)  & (spatial_input  != 0.0)
        physical_mask  = ~np.isnan(feats)

        return (
            torch.FloatTensor(spatial_input),
            torch.FloatTensor(physical_input),
            torch.BoolTensor(spatial_mask),
            torch.BoolTensor(physical_mask),
            buoy_id,
            ts.strftime("%Y%m%d%H"),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_bias_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        logging.warning("Bias file not found: %s", file_path)
        return pd.DataFrame()
    df = pd.read_excel(file_path).rename(columns={"Buoy_ID": "id", "Time": "time"})
    df["time"] = (
        df["time"].astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )
    df["time"] = pd.to_datetime(df["time"], format="%Y%m%d%H", errors="coerce")
    for col in ("Pred_Bias_Hs", "Pred_Bias_Tm"):
        if col not in df.columns:
            df[col] = 0.0
    return df[["id", "time", "Pred_Bias_Hs", "Pred_Bias_Tm"]]


def _normalize_time(df: pd.DataFrame, months: list[int]) -> pd.DataFrame:
    df["time"] = (
        df["time"].astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .apply(lambda x: str(int(float(x))) if "E" in x.upper() else x)
    )
    df["time"] = pd.to_datetime(df["time"], format="%Y%m%d%H", errors="coerce").dt.floor("h")
    df = df[df["time"].dt.month.isin(months)]
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[
            logging.FileHandler(
                RESULT_DIR / "log_aus_transformer_pred_additive.log",
                encoding="utf-8",
            ),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    # --- Load model ---
    if not PredictionConfig.model_path.exists():
        logger.error("Model file not found: %s", PredictionConfig.model_path)
        return

    model = TransformerWavePredictor().to(PredictionConfig.device)
    model.load_state_dict(
        torch.load(str(PredictionConfig.model_path), map_location=PredictionConfig.device)
    )
    model.eval()

    with open(PredictionConfig.scaler_path, "rb") as fh:
        scaler: ScenarioScaler = pickle.load(fh)
    logger.info("Model loaded  (device: %s)", PredictionConfig.device)

    # --- Load bias predictions ---
    bias_df = load_bias_data(PredictionConfig.bias_file)
    gc.collect()

    # --- Load input data ---
    if not PredictionConfig.data_files["depth"].exists():
        logger.error("Depth file not found: %s", PredictionConfig.data_files["depth"])
        return

    logger.info("Reading input CSVs …")
    depth_df = pd.read_csv(PredictionConfig.data_files["depth"], dtype={"time": str})
    wind_df  = pd.read_csv(PredictionConfig.data_files["wind"],  dtype={"time": str})

    for d in (depth_df, wind_df):
        d.rename(columns={"Time": "time", "ID": "id"}, inplace=True)
        d.columns = [c.lower() if c in ("TIME", "ID") else c for c in d.columns]

    depth_df = _normalize_time(depth_df, PredictionConfig.test_months)
    wind_df  = _normalize_time(wind_df,  PredictionConfig.test_months)

    input_df = depth_df.merge(wind_df, on=["id", "time"], how="inner")
    del wind_df; gc.collect()

    # --- Build dataset & dataloader ---
    logger.info("Building dataset …")
    depth_for_coords = pd.read_csv(
        PredictionConfig.data_files["depth"], dtype={"time": str}
    ).rename(columns={"Time": "time", "ID": "id"})
    coord_sys = SpatialCoordManager.load_coordinates(depth_for_coords, PredictionConfig.test_months)
    del depth_for_coords, depth_df; gc.collect()

    ds     = PredictionDataset(input_df, coord_sys, scaler)
    loader = DataLoader(ds, batch_size=PredictionConfig.batch_size, num_workers=0)
    del input_df; gc.collect()

    # --- Inference ---
    results: dict = {"id": [], "time": [], "raw_hs": [], "raw_tm": [], "depth": []}
    logger.info("Running inference …")
    with torch.no_grad():
        for batch in tqdm(loader):
            spatial, physical, s_mask, p_mask, ids, times = batch
            out = model(
                spatial.to(PredictionConfig.device),
                physical.to(PredictionConfig.device),
                s_mask.to(PredictionConfig.device),
                p_mask.to(PredictionConfig.device),
            ).cpu().numpy()

            results["id"].extend(ids)
            results["time"].extend(times)
            results["raw_hs"].append(scaler.inverse_transform_targets(out[:, 0, :], "hs"))
            results["raw_tm"].append(scaler.inverse_transform_targets(out[:, 1, :], "tm"))
            results["depth"].append(spatial.numpy()[:, 0, :])

    del ds, loader; torch.cuda.empty_cache(); gc.collect()

    # --- Post-processing ---
    logger.info("Applying bias correction and physical limits …")
    base_df  = pd.DataFrame({"id": results["id"], "time": pd.to_datetime(results["time"], format="%Y%m%d%H")})
    arr_hs   = np.concatenate(results["raw_hs"],  axis=0)
    arr_tm   = np.concatenate(results["raw_tm"],  axis=0)
    arr_depth = np.concatenate(results["depth"], axis=0)
    del results; gc.collect()

    arr_hs   = np.maximum(arr_hs, 0.0)
    arr_tm   = np.maximum(arr_tm, 0.0)

    final_hs = arr_hs.copy()
    final_tm = arr_tm.copy()

    if not bias_df.empty:
        merged = base_df.merge(bias_df, on=["id", "time"], how="left")
        bhs    = merged["Pred_Bias_Hs"].fillna(0).values[:, None].astype(np.float32)
        btm    = merged["Pred_Bias_Tm"].fillna(0).values[:, None].astype(np.float32)
        final_hs = np.maximum(arr_hs - bhs, 0.0)
        final_tm = np.maximum(arr_tm - btm, 0.0)
        del merged, bias_df, bhs, btm; gc.collect()

    # Breaking limit: Hs ≤ 0.78 × depth
    final_hs = np.maximum(np.minimum(final_hs, arr_depth * 0.78), 0.0)

    # --- Save ---
    save_path = RESULT_DIR / "Aus_Final_Pred_2021_Additive_Limit.xlsx"
    logger.info("Saving → %s", save_path)
    cols_hs   = [f"hs_{i:03d}" for i in PredictionConfig.depth_indices]
    cols_tm   = [f"tm_{i:03d}" for i in PredictionConfig.depth_indices]

    try:
        with pd.ExcelWriter(str(save_path), engine="openpyxl") as writer:
            base_df.to_excel(writer, sheet_name="Meta", index=False)
            for data, cols, sheet in [
                (final_hs, cols_hs, "Hs_Corrected"),
                (final_tm, cols_tm, "Tm_Corrected"),
                (arr_hs,   cols_hs, "Hs_Raw"),
                (arr_tm,   cols_tm, "Tm_Raw"),
            ]:
                pd.concat([base_df, pd.DataFrame(data, columns=cols)], axis=1).to_excel(
                    writer, sheet_name=sheet, index=False
                )
    except Exception as exc:
        logger.error("Save failed: %s", exc)

    logger.info("Done.")


if __name__ == "__main__":
    main()
