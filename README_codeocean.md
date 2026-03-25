# SWAN-TC: A Hybrid Transformer–CatBoost Framework for Nearshore Wave Profile Prediction

GitHub: https://github.com/daln45/SWAN-TC

## Abstract

We present **SWAN-TC**, a hybrid deep-learning pipeline for predicting
cross-shore wave profiles (significant wave height *H*s and mean wave period
*T*m) at 64 nearshore grid points from ERA5 offshore forcing and SWAN
bathymetry profiles.  The framework couples a Transformer encoder (**SWAN-T**)
with a CatBoost gradient-boosted bias corrector (**SWAN-C**) and enforces a
physical wave-breaking ceiling (*H*s ≤ 0.78 × depth) during inference.

## Model Overview

| Component | Type | Role |
|-----------|------|------|
| **SWAN-T** | Transformer encoder | Predicts *H*s / *T*m at 64 cross-shore depth nodes |
| **SWAN-C** | CatBoost multi-output regressor | Learns bias Δ*H*s, Δ*T*m from SWAN-T vs. observations |
| **SWAN-TC** | Inference pipeline | SWAN-T → SWAN-C correction → breaking limit |

SWAN-T is trained on 2020 UK SWAN model outputs plus 200 000 synthetic beach
profiles.  SWAN-C corrects residual biases using buoy observations.
Both models transfer directly to 12 Australian sites without retraining.

---

## Data Availability Note

Due to file-size restrictions, only the data required for figure reproduction
is included in this capsule.  The full dataset (raw SWAN outputs, ERA5 NetCDF,
training CSVs, prediction CSVs, ~8 GB total) is available on GitHub via
Git LFS: https://github.com/daln45/SWAN-TC

**Figures 1, 2, and 3 are fully reproducible** with the data in this capsule.
Figures 4 and 5 require large files not included here (see GitHub).

---

## Repository Structure

```
SWAN-TC/
|-- run                          # CodeOcean entry-point (Bash) — runs Fig 2 & Fig 3
|-- requirements.txt
|-- README.md                    # This file
|-- .gitattributes               # Git LFS tracking (*.pth, *.pkl, *.cbm, *.csv, *.xlsx, *.nc)
|
|-- code/
|   |-- SWAN-T/
|   |   `-- train.py             # Transformer training (data-efficiency experiments)
|   |-- SWAN-C/
|   |   |-- train.py             # CatBoost bias-correction training
|   |   `-- predict.py           # Bias inference for 2021–2023
|   |-- SWAN-TC/
|   |   `-- predict.py           # Full SWAN-TC inference pipeline
|   |-- preprocessing/
|   |   |-- generate_profiles.py # Synthetic beach-profile generation
|   |   `-- sample_boundary.py   # Max-dissimilarity boundary sampling
|   |-- australia/
|   |   |-- predict_bias.py      # SWAN-C bias prediction (Australia)
|   |   `-- predict_transformer.py # SWAN-TC prediction (Australia)
|   `-- figures/
|       |-- fig1/  prepare_data.py  plot.py   # Study area & ERA5 overview        ✓ runnable
|       |-- fig2/  plot.py                    # SWAN-T data-efficiency comparison  ✓ runnable
|       |-- fig3/  plot.py                    # SWAN-C bias-correction evaluation  ✓ runnable
|       |-- fig4/  plot.py                    # SWAN-TC UK validation              [full dataset needed]
|       `-- fig5/  plot.py                    # SWAN-TC Australia transfer         [full dataset needed]
|
|-- data/                        # Input data included in this capsule
|   |-- fig1/
|   |   |-- processed_plot_data_3x3.pkl   # Pre-processed data for Fig 1 plot
|   |   |-- land_polygons.shp (+ sidecar) # Coastline shapefile
|   |   `-- (other supporting files)
|   |-- uk/
|   |   |-- swan_hs_2020.csv     # SWAN modelled Hs, 64-point profiles, 2020
|   |   `-- swan_tm_2020.csv     # SWAN modelled Tm, 64-point profiles, 2020
|   |-- fig2/bot/
|   |   `-- {Fxs,Hrn,PBy,WBy}_waves.bot  # Bathymetry transects
|   `-- fig3/
|       `-- buoy_data_2021.xlsx  # In-situ buoy observations, 2021
|
|-- weights/                     # Pre-trained model files
|   |-- best_model_exp_5_100_percent.pth   # SWAN-T Transformer weights
|   |-- scaler_exp_5_100_percent.pkl       # SWAN-T input scaler
|   |-- best_multi_bias_model.cbm          # SWAN-C CatBoost weights
|   `-- scaler_multi_bias.pkl              # SWAN-C input scaler
|
`-- results/                     # Pre-computed outputs
    |-- swan_t/  results_exp_5_100_percent.xlsx, results_exp_7_100_real_only.xlsx
    `-- swan_c/  results_2020_TRAIN_EVAL.xlsx, results_2021_TEST_EVAL.xlsx
```

---

## Environment

```bash
pip install -r requirements.txt
```

Key packages: `torch`, `catboost`, `pandas`, `numpy`, `scikit-learn`,
`matplotlib`, `cartopy`, `geopandas`, `netCDF4`, `openpyxl`, `tqdm`.

Python ≥ 3.10 is required.

---

## Reproducing the Results

Click **Reproducible Run** on CodeOcean, or run locally:

```bash
bash run
```

This reproduces **Figures 1, 2, and 3**.

```bash
python code/figures/fig1/plot.py    # Fig 1 — study area overview
python code/figures/fig2/plot.py    # Fig 2 — SWAN-T data-efficiency
python code/figures/fig3/plot.py    # Fig 3 — SWAN-C bias correction
```

Figures 4 and 5 require the full dataset. Clone from GitHub and run
`git lfs pull`, then:

```bash
python code/figures/fig4/plot.py    # Fig 4 — SWAN-TC UK validation
python code/figures/fig5/plot.py    # Fig 5 — SWAN-TC Australia transfer
```

To re-run inference or retrain models, see the full instructions on
https://github.com/daln45/SWAN-TC

---

## Model Configuration

| Parameter | Value |
|-----------|-------|
| `d_model` | 256 |
| `nhead` | 8 |
| `num_layers` | 6 |
| `dim_feedforward` | 512 |
| `activation` | gelu |
| `spatial_points` | 64 |
| `grid_spacing` | 50 m |
| `depth_indices` | range(10, 200, 3) — 64 points |
| Physical input channels | 7 (Hs, wind speed, wind direction, MWD, MWP, ALPC, tide) |
| Breaking limit | *H*s ≤ 0.78 × depth |

---

## Citation

If you use this code or data, please cite the associated paper
(details to be added upon acceptance).

---

## License

MIT
