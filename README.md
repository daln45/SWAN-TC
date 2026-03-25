# SWAN-TC

SWAN-TC is a hybrid deep learning framework for nearshore wave profile
prediction, combining a Transformer-based spatial model (SWAN-T) with a
CatBoost bias-correction model (SWAN-C), with wave-breaking constraints
applied during inference.

## Architecture

```
ERA5 offshore forcing  +  SWAN bathymetry profiles
            |
          SWAN-T  -->  raw Hs / Tm profiles  (64 cross-shore points)
            |
          SWAN-C  -->  bias correction  (Delta_Hs, Delta_Tm) at each buoy / timestep
            |
           correction + breaking limit  (Hs <= 0.78 x depth)
            |
          SWAN-TC -->  final corrected Hs / Tm profiles
```

- **SWAN-T** - Transformer encoder; inputs: depth bathymetry profiles + physical forcing
  (Hs, Tp, Tm, wind speed/direction, tide); outputs: spatial Hs and Tm across 64 depth nodes.
- **SWAN-C** - CatBoost multi-output regressor; predicts systematic bias between SWAN-T and observations.
- **SWAN-TC** - Full pipeline: SWAN-T -> SWAN-C additive correction -> physical breaking limit.

---

## Repository Structure

```
SWAN-TC/
|-- run                          # CodeOcean entry-point (Bash)
|-- requirements.txt
|-- .gitattributes               # Git LFS rules for *.pth / *.pkl / *.cbm / *.csv / *.xlsx / *.nc
|
|-- code/
|   |-- SWAN-T/
|   |   `-- train.py             # Transformer training script (data efficiency experiments)
|   |-- SWAN-C/
|   |   |-- train.py             # CatBoost bias-correction training
|   |   `-- predict.py           # Generate bias predictions for 2021-2023
|   |-- SWAN-TC/
|   |   `-- predict.py           # Full SWAN-TC inference pipeline
|   |-- preprocessing/
|   |   |-- generate_profiles.py # Synthetic beach-profile generation
|   |   `-- sample_boundary.py   # Max-dissimilarity boundary sampling
|   |-- australia/
|   |   |-- predict_bias.py      # CatBoost bias prediction (Australia)
|   |   `-- predict_transformer.py # Transformer prediction (Australia)
|   `-- figures/
|       |-- fig1/  prepare_data.py  plot.py
|       |-- fig2/  plot.py
|       |-- fig3/  plot.py
|       |-- fig4/  plot.py
|       `-- fig5/  plot.py
|
|-- data/                        # All input data (tracked by Git LFS)
|   |-- uk/                      # UK SWAN model input (2020-2023)
|   |   |-- swan_depth_{year}.csv            # Depth profiles (64 nodes x all buoys)
|   |   |-- wind_wave_initial_data_{year}.csv # ERA5 offshore forcing
|   |   |-- swan_hs_2020.csv / swan_tm_2020.csv
|   |   |-- virtual_profile_{depth,hs,tm}.csv # Synthetic training profiles
|   |   |-- boundary_data_2020.csv
|   |   |-- buoy_data_{year}.xlsx            # In-situ buoy observations
|   |   `-- bias_data/           # SWAN bias fields for SWAN-C training
|   |-- fig1/                    # ERA5 NetCDF, buoy CSV, shapefiles, dot.txt
|   |-- fig2/bot/                # Cross-shore bathymetry (.bot) for 4 test buoys
|   |-- fig3/                    # Buoy observations for SWAN-C evaluation
|   |-- fig4_fig5/               # Buoy observations, WW3 baseline, Australia predictions
|   |   `-- ww3/                 # WW3 baseline xlsx (Jan/Jul, 2021-2023)
|   `-- aus/                     # Australia depth, ERA5, buoy data and bot files
|
|-- weights/                     # Pre-trained model files (tracked by Git LFS)
|   |-- best_model_exp_5_100_percent.pth   # SWAN-T weights
|   |-- scaler_exp_5_100_percent.pkl       # SWAN-T input scaler
|   |-- best_multi_bias_model.cbm          # SWAN-C weights
|   `-- scaler_multi_bias.pkl              # SWAN-C input scaler
|
`-- results/                     # Pre-computed outputs (tracked by Git LFS)
    |-- swan_t/                  # SWAN-T training results per data-efficiency experiment
    |-- swan_c/                  # SWAN-C bias evaluation (train 2020, test 2021-2023)
    `-- swan_tc/                 # SWAN-TC final predictions (Final_Pred_{year}_{Hs,Tm}_{Raw,Corrected}.csv)
```

---

## Environment

Python 3.10+ with the packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `catboost`, `pandas`, `numpy`, `scikit-learn`,
`matplotlib`, `cartopy`, `openpyxl`, `tqdm`.

---

## Data

All input data are included in this repository via **Git LFS** (`.csv`, `.xlsx`, `.nc`,
`.pth`, `.pkl`, `.cbm`).  After cloning, run:

```bash
git lfs pull
```

to download all large files.  No external data downloads are required to reproduce
the figures.  The full data layout is described in the Repository Structure section above.

---

## How to Reproduce

### 0  Install dependencies

```bash
pip install -r requirements.txt
```

### 1  Generate synthetic beach profiles  (preprocessing)

```bash
python code/preprocessing/generate_profiles.py
```

Output: `data/profiles/standard/` and `data/profiles/dune/`

### 2  Sample boundary conditions  (preprocessing)

```bash
python code/preprocessing/sample_boundary.py \
       --input  data/uk/wind_wave_initial_data_2020.csv \
       --output data/uk/boundary_data_2020.csv
```

### 3  Train SWAN-T  (Transformer)

```bash
python code/SWAN-T/train.py
```

Trained weights are saved to `./results_data_efficiency_*/` relative to the script.
Copy the final `.pth` and `.pkl` files to `weights/` to use with SWAN-TC inference.
Pre-trained weights are already provided in `weights/`.

### 4  Train SWAN-C  (CatBoost bias corrector)

```bash
python code/SWAN-C/train.py
```

Reads from `data/uk/`.  Pre-trained weights are already provided in `weights/`.

### 5  Generate SWAN-C bias predictions

```bash
python code/SWAN-C/predict.py
```

Output: `results/swan_c/results_{year}_PREDICTION.xlsx`.
Pre-computed results are already provided in `results/swan_c/`.

### 6  UK validation — generate SWAN-TC predictions

```bash
python code/SWAN-TC/predict.py
```

Reads model weights from `weights/` and bias results from `results/swan_c/`.
Output: `results/swan_tc/Final_Pred_{year}_{Hs,Tm}_{Raw,Corrected}.csv`.
Pre-computed results are already provided in `results/swan_tc/`.

### 7  Australia — predict bias  (transfer learning)

```bash
python code/australia/predict_bias.py
```

Output: `results/aus_bias_predictions/results_2021_01_Aus_Pred.xlsx`

### 8  Australia — SWAN-T + bias correction

```bash
python code/australia/predict_transformer.py
```

Output: `results/aus_transformer/Aus_Final_Pred_2021_Additive_Limit.xlsx`

### 9  Reproduce figures

All figure scripts resolve data paths automatically via `Path(__file__)` —
they can be run from any working directory.

```bash
python code/figures/fig1/prepare_data.py
python code/figures/fig1/plot.py
python code/figures/fig2/plot.py
python code/figures/fig3/plot.py
python code/figures/fig4/plot.py
python code/figures/fig5/plot.py
```

Figures are written to the same directory as each script (or `results/figures/`
if so configured inside the script).

---

## Model Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 256 |
| nhead | 8 |
| num_layers | 6 |
| dim_feedforward | 512 |
| activation | gelu |
| spatial_points | 64 |
| grid_spacing | 50 m |
| depth_indices | range(10, 200, 3)  — 64 points |
| Physical channels | 7 (swh, wind_speed, wind_direction, mwd, mwp, alpc, tide) |

---

## Citation

If you use this code, please cite the associated paper (details to be added upon acceptance).

---

## License

MIT
