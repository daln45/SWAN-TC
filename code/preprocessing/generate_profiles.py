# -*- coding: utf-8 -*-
"""
Generate synthetic beach profiles for SWAN-T training.

Two profile types are created:
  1. Standard Bruun-Rule profiles  (n_samples=30)
  2. Dune-backed beach profiles     (n_dune_samples=20)

All profiles are written as SWAN bathymetry (.bot) files and
corresponding depth-fraction (.f) files consumed by the
training pipeline.

Usage
-----
    python generate_profiles.py

Output is placed in ``data/profiles/`` relative to this file.
"""

from __future__ import annotations

import numpy as np
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.parent.parent          # repo root
OUTPUT_DIR = BASE_DIR / "data" / "profiles"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Grid configuration  (must match training settings)
# ---------------------------------------------------------------------------

GRID_SPACING   = 50          # metres between grid nodes
N_GRID_POINTS  = 200         # total nodes in cross-shore direction
DEPTH_INDICES  = list(range(10, 200, 3))[:64]   # 64 active nodes

# Profile parameter ranges
SHORE_DIST_MIN = 3_000       # metres  (shallowest node distance from shore)
SHORE_DIST_MAX = 7_000       # metres

# ---------------------------------------------------------------------------
# Bruun-Rule profile helpers
# ---------------------------------------------------------------------------


def _equilibrium_depth(x: np.ndarray, A: float) -> np.ndarray:
    """Dean's equilibrium beach profile:  h(x) = A * x^(2/3)."""
    return A * np.power(np.maximum(x, 0.0), 2.0 / 3.0)


def _bruun_profile(
    n_points: int,
    grid_spacing: float,
    A: float,
    x_offset: float,
) -> np.ndarray:
    """Return a 1-D depth array (positive downward) from shore to offshore."""
    x = np.arange(n_points) * grid_spacing + x_offset
    return _equilibrium_depth(x, A)


# ---------------------------------------------------------------------------
# Standard profile generation
# ---------------------------------------------------------------------------

def generate_ideal_beach_profiles(
    n_samples: int = 30,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """
    Generate *n_samples* Bruun-Rule equilibrium beach profiles.

    Parameters
    ----------
    n_samples :
        Number of profiles to generate.
    rng :
        Optional NumPy random generator for reproducibility.

    Returns
    -------
    list of 1-D arrays, each of length N_GRID_POINTS
    """
    if rng is None:
        rng = np.random.default_rng(42)

    profiles: list[np.ndarray] = []
    for _ in range(n_samples):
        A        = rng.uniform(0.05, 0.25)          # dean sediment param
        x_offset = rng.uniform(SHORE_DIST_MIN, SHORE_DIST_MAX)
        prof     = _bruun_profile(N_GRID_POINTS, GRID_SPACING, A, x_offset)
        profiles.append(prof)

    return profiles


# ---------------------------------------------------------------------------
# Dune profile generation
# ---------------------------------------------------------------------------

def _add_dune(depth: np.ndarray, dune_height: float, dune_width_nodes: int) -> np.ndarray:
    """
    Superimpose a Gaussian dune feature on the first *dune_width_nodes* nodes
    of *depth*.  Dune toe → crest → back slope is approximated by the
    positive half of a Gaussian.
    """
    prof = depth.copy()
    sigma = dune_width_nodes / 3.0
    x_nodes = np.arange(dune_width_nodes)
    dune = dune_height * np.exp(-0.5 * ((x_nodes - 0) / sigma) ** 2)
    # dune raises the bed (reduces depth)
    prof[:dune_width_nodes] -= dune
    return np.maximum(prof, 0.0)   # depth cannot be negative


def generate_dune_beach_profiles(
    n_dune_samples: int = 20,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """
    Generate *n_dune_samples* dune-backed beach profiles.

    Parameters
    ----------
    n_dune_samples :
        Number of dune profiles to generate.
    rng :
        Optional NumPy random generator for reproducibility.

    Returns
    -------
    list of 1-D arrays, each of length N_GRID_POINTS
    """
    if rng is None:
        rng = np.random.default_rng(123)

    profiles: list[np.ndarray] = []
    for _ in range(n_dune_samples):
        A            = rng.uniform(0.05, 0.20)
        x_offset     = rng.uniform(SHORE_DIST_MIN, SHORE_DIST_MAX)
        base         = _bruun_profile(N_GRID_POINTS, GRID_SPACING, A, x_offset)
        dune_height  = rng.uniform(1.0, 5.0)           # metres
        dune_width   = int(rng.uniform(3, 10))          # grid nodes
        prof         = _add_dune(base, dune_height, dune_width)
        profiles.append(prof)

    return profiles


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def _write_bot_file(path: Path, depth: np.ndarray) -> None:
    """Write a SWAN .bot bathymetry file (one depth per line)."""
    with open(path, "w") as fh:
        for d in depth:
            fh.write(f"{d:.4f}\n")


def _write_f_file(path: Path, depth: np.ndarray) -> None:
    """
    Write a condensed depth-fraction (.f) file containing only the
    active *DEPTH_INDICES* nodes.
    """
    active = depth[DEPTH_INDICES]
    with open(path, "w") as fh:
        fh.write(" ".join(f"{v:.4f}" for v in active) + "\n")


def process_synthetic_to_f_files(
    profiles: list[np.ndarray],
    prefix: str,
    out_dir: Path,
) -> None:
    """
    Persist *profiles* as paired .bot + .f files inside *out_dir*.

    File names follow the convention ``<prefix>_<zero-padded-index>.bot/.f``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_digits = len(str(len(profiles)))
    for i, prof in enumerate(profiles):
        stem = f"{prefix}_{i:0{n_digits}d}"
        _write_bot_file(out_dir / f"{stem}.bot", prof)
        _write_f_file(out_dir / f"{stem}.f",   prof)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(0)

    print("Generating standard Bruun-Rule profiles …")
    standard_profiles = generate_ideal_beach_profiles(n_samples=30, rng=rng)
    process_synthetic_to_f_files(standard_profiles, prefix="standard", out_dir=OUTPUT_DIR / "standard")
    print(f"  Wrote {len(standard_profiles)} profiles → {OUTPUT_DIR / 'standard'}")

    print("Generating dune-backed profiles …")
    dune_profiles = generate_dune_beach_profiles(n_dune_samples=20, rng=rng)
    process_synthetic_to_f_files(dune_profiles, prefix="dune", out_dir=OUTPUT_DIR / "dune")
    print(f"  Wrote {len(dune_profiles)} profiles → {OUTPUT_DIR / 'dune'}")

    print("Done.")


if __name__ == "__main__":
    main()
