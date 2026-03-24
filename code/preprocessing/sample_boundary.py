# -*- coding: utf-8 -*-
"""
Max-Dissimilarity Sampling (MDS) for SWAN boundary conditions.

Selects 2000 representative boundary-condition vectors from a larger ERA5
input dataset so that the training set spans the full parameter space as
uniformly as possible.

Tide filtering is applied before sampling to remove physically unrealistic
conditions (|tide| > 10 m → dropped).

Usage
-----
    python sample_boundary.py --input data/era5_raw.csv --output data/boundary_data_2020.csv

Output columns
--------------
    time, swh, wind_speed, wind_direction, mwp, mwd, tide, sin_wd, cos_wd,
    sin_mwd, cos_mwd
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (defaults)
# ---------------------------------------------------------------------------

BASE_DIR   = Path(__file__).parent.parent.parent
INPUT_CSV  = BASE_DIR / "data" / "era5_raw.csv"
OUTPUT_CSV = BASE_DIR / "data" / "boundary_data_2020.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SAMPLES           = 2_000
TIDE_THRESHOLD      = (-10.0, 10.0)   # metres  — drop records outside range
TIDE_MODE           = "drop"          # "drop" | "clip"

FEATURE_COLS = [
    "swh", "wind_speed", "mwp", "tide",
    "sin_wd", "cos_wd", "sin_mwd", "cos_mwd",
]


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _add_trig_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sin/cos encodings for wind_direction and mwd."""
    df = df.copy()
    if "wind_direction" in df.columns:
        wd_rad = np.deg2rad(df["wind_direction"])
        df["sin_wd"] = np.sin(wd_rad)
        df["cos_wd"] = np.cos(wd_rad)
    if "mwd" in df.columns:
        mwd_rad = np.deg2rad(df["mwd"])
        df["sin_mwd"] = np.sin(mwd_rad)
        df["cos_mwd"] = np.cos(mwd_rad)
    return df


def _apply_tide_threshold(
    df: pd.DataFrame,
    lo: float = TIDE_THRESHOLD[0],
    hi: float = TIDE_THRESHOLD[1],
    mode: str = TIDE_MODE,
) -> pd.DataFrame:
    """
    Filter rows by tide level.

    Parameters
    ----------
    df :
        Input dataframe.
    lo, hi :
        Lower and upper tide bounds (metres).
    mode :
        ``"drop"``  — remove rows outside [lo, hi].
        ``"clip"``  — clamp tide values to [lo, hi].
    """
    if "tide" not in df.columns:
        return df
    mask = (df["tide"] >= lo) & (df["tide"] <= hi)
    if mode == "drop":
        n_removed = (~mask).sum()
        df = df[mask].reset_index(drop=True)
        logger.info("Tide filter: removed %d rows  (|tide| > %.1f m)", n_removed, max(abs(lo), abs(hi)))
    elif mode == "clip":
        df["tide"] = df["tide"].clip(lower=lo, upper=hi)
        logger.info("Tide filter: clipped to [%.1f, %.1f] m", lo, hi)
    return df


def _robust_scale(X: np.ndarray) -> np.ndarray:
    """Apply RobustScaler along feature axis."""
    scaler = RobustScaler()
    return scaler.fit_transform(X)


# ---------------------------------------------------------------------------
# Max-dissimilarity sampling
# ---------------------------------------------------------------------------

def max_dissimilarity_sampling(
    X_scaled: np.ndarray,
    n_samples: int = N_SAMPLES,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Greedy max-dissimilarity (farthest-point) sampling.

    Iteratively selects the candidate point that maximises the minimum
    Euclidean distance to the already-selected set.

    Parameters
    ----------
    X_scaled :
        Pre-scaled feature matrix of shape *(N, D)*.
    n_samples :
        Number of points to select.
    rng :
        Optional random generator (used to pick the first seed point).

    Returns
    -------
    selected_indices : ndarray of shape *(n_samples,)*
        Row indices of selected points in the original order.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    N = X_scaled.shape[0]
    n_samples = min(n_samples, N)

    # Start from a random seed
    seed = int(rng.integers(N))
    selected: list[int] = [seed]
    # min-distance from each point to the selected set
    min_dist = np.full(N, np.inf)

    logger.info("Running MDS: %d → %d samples …", N, n_samples)
    for step in range(1, n_samples):
        # Update min_dist using the most recently added point
        last = X_scaled[selected[-1]]
        dist_to_last = np.linalg.norm(X_scaled - last, axis=1)
        min_dist = np.minimum(min_dist, dist_to_last)
        min_dist[selected[-1]] = -np.inf  # already selected

        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)

        if step % 200 == 0 or step == n_samples - 1:
            logger.info("  %d / %d", step, n_samples)

    return np.array(selected, dtype=int)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(input_csv: Path, output_csv: Path, n_samples: int = N_SAMPLES) -> None:
    logger.info("Loading %s", input_csv)
    df = pd.read_csv(input_csv)

    # Trig features
    df = _add_trig_features(df)

    # Tide filtering
    df = _apply_tide_threshold(df)

    # Check all feature columns exist
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    # Scale
    X_raw    = df[FEATURE_COLS].values.astype(np.float64)
    X_scaled = _robust_scale(X_raw)

    # MDS selection
    idx = max_dissimilarity_sampling(X_scaled, n_samples=n_samples)

    sampled = df.iloc[idx].copy()
    sampled = sampled.reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(output_csv, index=False)
    logger.info("Saved %d samples → %s", len(sampled), output_csv)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Max-dissimilarity boundary sampling")
    p.add_argument(
        "--input",  type=Path, default=INPUT_CSV,
        help="Path to raw ERA5 CSV  (default: %(default)s)",
    )
    p.add_argument(
        "--output", type=Path, default=OUTPUT_CSV,
        help="Destination CSV path  (default: %(default)s)",
    )
    p.add_argument(
        "--n-samples", type=int, default=N_SAMPLES,
        help="Number of samples to select  (default: %(default)s)",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run(input_csv=args.input, output_csv=args.output, n_samples=args.n_samples)
