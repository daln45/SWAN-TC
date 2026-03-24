# -*- coding: utf-8 -*-
"""
Figure 1 – Data preparation
Process UK buoy metadata, observation data, ERA5 reanalysis, SWAN model outputs,
and SWAN-TC predictions into a single consolidated pickle file for plotting.
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.metrics import mean_absolute_error
from shapely.geometry import MultiLineString


# =========================
# Config
# =========================
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "fig1"

FILE_BUOY_META      = DATA_DIR / "uk_buoy.xlsx"
FILE_BUOY_DATA      = DATA_DIR / "buoy_data_2021.csv"
FILE_ERA5_NC        = DATA_DIR / "hs_2021.nc"
FILE_SWAN_DATA      = DATA_DIR / "swan_hs_2021.csv"
FILE_BOUNDARY_DATA  = DATA_DIR / "boundary_data_2021.csv"
FILE_PRED_EXCEL     = DATA_DIR / "results_2021_PREDICTION.xlsx"
FILE_POINTS_DOT     = DATA_DIR / "dot.txt"
SHAPEFILE_PATH      = DATA_DIR / "land_polygons.shp"
OUTPUT_DATA_FILE    = DATA_DIR / "processed_plot_data_3x3.pkl"

SHEET_PRED_RAW  = "Sheet1"
SNAPSHOT_TIME   = "2021010106"
TARGET_BUOY_DETAIL = "WBy_waves"

ORDERED_BUOY_IDS = [
    "Bdf_waves", "BkB_waves", "Bos_waves", "ChP_waves", "Clv_waves",
    "Csl_waves", "DlP_waves", "Dwl_waves", "Flk_waves", "Fxs_waves",
    "GdS_waves", "HBy_waves", "HgI_waves", "HgP_waves", "Hpg_waves",
    "Hrn_waves", "LoB_waves", "Lym_waves", "McB_waves", "Mhd_waves",
    "Mlf_waves", "Nbg_waves", "PBy_waves", "Plv_waves", "Pnz_waves",
    "Prp_waves", "RhF_waves", "Rst_waves", "Rye_waves", "SBy_waves",
    "Sca_waves", "SdP_waves", "Sfd_waves", "Spn_waves", "StB_waves",
    "TnP_waves", "Wan_waves", "WBy_waves", "Wey_waves", "Wtb_waves",
    "Bdf_waves2", "BkB_waves2", "Bos_waves2", "ChP_waves2", "Clv_waves2",
    "Csl_waves2", "DlP_waves2", "Dwl_waves2", "Flk_waves2", "Fxs_waves2",
]

BUOY_DISTANCES = {
    "Hrn_waves": 7015.19, "Fxs_waves": 3308.10,
    "PBy_waves": 5294.04, "WBy_waves": 1198.36,
    "Bos_waves": 2500.00, "HgI_waves": 4100.00,
    "ChP_waves": 3200.00, "Mlf_waves": 5800.00,
    "BkB_waves": 1900.00, "Clv_waves": 6100.00,
    "Csl_waves": 2750.00, "Dwl_waves": 4300.00,
    "Flk_waves": 3600.00, "GdS_waves": 4900.00,
    "Hpg_waves": 2200.00, "LoB_waves": 5000.00,
    "Nbg_waves": 3800.00, "Plv_waves": 4600.00,
    "Pnz_waves": 3100.00, "Prp_waves": 2600.00,
    "RhF_waves": 4100.00, "Rst_waves": 3400.00,
    "SBy_waves": 4700.00, "Sca_waves": 3900.00,
    "StB_waves": 2800.00, "Wtb_waves": 5300.00,
    "Bdf_waves": 2300.00, "DlP_waves": 4400.00,
    "HBy_waves": 3300.00, "HgP_waves": 4000.00,
    "Lym_waves": 2100.00, "McB_waves": 3700.00,
    "Mhd_waves": 4200.00, "Rye_waves": 2900.00,
    "SdP_waves": 4800.00, "Sfd_waves": 3500.00,
    "Spn_waves": 2400.00, "TnP_waves": 4500.00,
    "Wan_waves": 3000.00,
}

TEST_BUOYS = ["Hrn_waves", "Fxs_waves", "PBy_waves", "WBy_waves"]


# =========================
# Utilities
# =========================
def load_era5_corrected(nc_path):
    try:
        ds = xr.open_dataset(nc_path)
        return ds
    except Exception as e:
        print(f"Error loading ERA5: {e}")
        return None


def get_closest_column(columns, prefix, target_idx):
    best_col = None
    min_diff = float('inf')
    pattern = re.compile(rf"^{prefix}_?(\d+)$")
    for col in columns:
        match = pattern.match(col)
        if match:
            idx = int(match.group(1))
            diff = abs(idx - target_idx)
            if diff < min_diff:
                min_diff = diff
                best_col = col
    return best_col


def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def get_profile_data(row_series, columns, prefix, cell_size=50):
    dists, vals = [], []
    pattern = re.compile(rf"^{prefix}_?(\d+)$")
    for col in columns:
        match = pattern.match(col)
        if match:
            idx = int(match.group(1))
            d_km = idx * cell_size / 1000.0
            val = row_series[col]
            dists.append(d_km)
            vals.append(val)
    if not dists:
        return [], []
    sorted_pairs = sorted(zip(dists, vals))
    return [p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs]


def load_buoy_coords():
    try:
        df = pd.read_excel(FILE_BUOY_META)
        id_map = {row['id']: row for _, row in df.iterrows()}
        coords = {}
        for buoy_id in BUOY_DISTANCES.keys():
            if buoy_id in id_map:
                row = id_map[buoy_id]
                coords[buoy_id] = {'lat': row['lat'], 'lon': row['lon']}
        return coords
    except Exception:
        return {}


def load_real_obs(buoy_id):
    try:
        df = pd.read_csv(FILE_BUOY_DATA)
        df_buoy = df[df['id'] == buoy_id].copy()
        if df_buoy.empty:
            return None
        df_buoy['Time'] = pd.to_datetime(df_buoy['time'], format='%Y%m%d%H', errors='coerce')
        df_buoy['hs'] = pd.to_numeric(df_buoy['hs'], errors='coerce')
        df_buoy = df_buoy[(df_buoy['Time'] >= '2021-01-01') & (df_buoy['Time'] < '2022-01-01')]
        return df_buoy[['Time', 'hs']].set_index('Time').dropna()
    except Exception:
        return None


def load_boundary_value(buoy_id, target_time_str):
    try:
        df_bound = pd.read_csv(FILE_BOUNDARY_DATA)
        df_bound['time'] = df_bound['time'].astype(str)
        row = df_bound[(df_bound['id'] == buoy_id) & (df_bound['time'] == target_time_str)]
        if not row.empty:
            return float(row.iloc[0]['swh'])
        return np.nan
    except Exception as e:
        print(f"Error loading boundary file: {e}")
        return np.nan


def get_era5_mae(buoy_name, lat, lon, obs_df):
    try:
        ds = load_era5_corrected(FILE_ERA5_NC)
        if ds is None:
            return np.nan, np.nan, np.nan, np.nan
        var_name = 'swh' if 'swh' in ds else 'hs'

        lat_min_s = min(lat - 1.0, lat + 1.0)
        lat_max_s = max(lat - 1.0, lat + 1.0)
        lon_min_s = min(lon - 1.0, lon + 1.0)
        lon_max_s = max(lon - 1.0, lon + 1.0)

        ds_sub = ds.sel(
            longitude=slice(lon_min_s, lon_max_s),
            latitude=slice(lat_max_s, lat_min_s)
        )
        if ds_sub.latitude.size == 0:
            ds_sub = ds.sel(
                longitude=slice(lon_min_s, lon_max_s),
                latitude=slice(lat_min_s, lat_max_s)
            )
        if ds_sub.latitude.size == 0 or ds_sub.longitude.size == 0:
            return np.nan, np.nan, np.nan, np.nan

        sample = ds_sub[var_name].isel(time=0).values
        lons_grid, lats_grid = np.meshgrid(ds_sub.longitude.values, ds_sub.latitude.values)
        valid_mask = ~np.isnan(sample)
        if not np.any(valid_mask):
            return np.nan, np.nan, np.nan, np.nan

        valid_lons = lons_grid[valid_mask]
        valid_lats = lats_grid[valid_mask]
        dists_deg = np.sqrt((valid_lons - lon) ** 2 + (valid_lats - lat) ** 2)
        min_idx = np.argmin(dists_deg)
        nearest_sea_lon = valid_lons[min_idx]
        nearest_sea_lat = valid_lats[min_idx]
        dist_km = dists_deg[min_idx] * 111.32

        era5_series = ds[var_name].sel(longitude=nearest_sea_lon, latitude=nearest_sea_lat)
        df_era5 = era5_series.to_dataframe().reset_index().set_index('time')
        if df_era5.index.duplicated().any():
            df_era5 = df_era5[~df_era5.index.duplicated(keep='first')]

        common = obs_df.index.intersection(df_era5.index)
        if len(common) == 0:
            return np.nan, np.nan, np.nan, np.nan

        mae = calculate_mae(obs_df.loc[common, 'hs'], df_era5.loc[common, var_name])
        ds.close()
        return mae, nearest_sea_lon, nearest_sea_lat, dist_km
    except Exception as e:
        print(f"ERA5 MAE Error for {buoy_name}: {e}")
        return np.nan, np.nan, np.nan, np.nan


def get_model_mae(buoy_id, obs_df, df_pred, target_idx):
    try:
        df_p = df_pred[df_pred['id'] == buoy_id].copy()
        if df_p.empty:
            return np.nan
        df_p['time'] = pd.to_datetime(df_p['time'])
        df_p = df_p.set_index('time')
        col_name = get_closest_column(df_p.columns, "hs", target_idx)
        if not col_name:
            return np.nan
        common = obs_df.index.intersection(df_p.index)
        if len(common) == 0:
            return np.nan
        return calculate_mae(obs_df.loc[common, 'hs'], df_p.loc[common, col_name])
    except Exception:
        return np.nan


def analyze_coast_to_era5_distance(extent):
    print("Analyzing coastline to nearest ERA5 grid point distances...")
    try:
        ds = load_era5_corrected(FILE_ERA5_NC)
        if ds is None:
            return [], [], []
        var_name = 'swh' if 'swh' in ds else 'hs'

        min_lon, max_lon, min_lat, max_lat = extent
        ds_sub = ds.sel(
            longitude=slice(min_lon - 1.0, max_lon + 1.0),
            latitude=slice(max_lat + 1.0, min_lat - 1.0)
        )

        sample = ds_sub[var_name].isel(time=0)
        valid_mask = ~np.isnan(sample.values)
        lons_grid, lats_grid = np.meshgrid(ds_sub.longitude.values, ds_sub.latitude.values)
        valid_era5_lons = lons_grid[valid_mask]
        valid_era5_lats = lats_grid[valid_mask]

        if len(valid_era5_lons) == 0:
            return [], [], []

        mean_lat_rad = np.radians(np.mean(valid_era5_lats))
        lon_scale = np.cos(mean_lat_rad)
        tree = cKDTree(np.column_stack((valid_era5_lons * lon_scale, valid_era5_lats)))

        gdf_land = gpd.read_file(SHAPEFILE_PATH)
        bbox = (min_lon, min_lat, max_lon, max_lat)
        gdf_clip = gdf_land.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]

        coast_points_list = []
        for geom in gdf_clip.geometry:
            boundary = geom.boundary
            if boundary.is_empty:
                continue
            if isinstance(boundary, MultiLineString):
                lines = boundary.geoms
            else:
                lines = [boundary]
            for line in lines:
                xx, yy = line.coords.xy
                pts = np.column_stack((xx, yy))
                mask = (
                    (pts[:, 0] >= min_lon) & (pts[:, 0] <= max_lon) &
                    (pts[:, 1] >= min_lat) & (pts[:, 1] <= max_lat)
                )
                coast_points_list.append(pts[mask])

        if not coast_points_list:
            return [], [], []
        all_coast_points = np.vstack(coast_points_list)
        dists_deg, indices = tree.query(
            np.column_stack((all_coast_points[:, 0] * lon_scale, all_coast_points[:, 1]))
        )
        dists_km = dists_deg * 111.32

        unique_indices = np.unique(indices)
        selected_era5_lons = valid_era5_lons[unique_indices]
        selected_era5_lats = valid_era5_lats[unique_indices]
        ds.close()
        return dists_km, selected_era5_lons, selected_era5_lats
    except Exception as e:
        print(f"Error in coast-to-ERA5 analysis: {e}")
        return [], [], []


def load_fixed_coast_metadata(buoy_id):
    print(f"Loading fixed metadata for {buoy_id}...")
    try:
        if buoy_id not in ORDERED_BUOY_IDS:
            print(f"Warning: {buoy_id} not in ORDERED_BUOY_IDS.")
            return None, None, None

        idx = ORDERED_BUOY_IDS.index(buoy_id)

        if not os.path.exists(FILE_POINTS_DOT):
            print(f"Error: {FILE_POINTS_DOT} not found.")
            return None, None, None

        df_dot = pd.read_csv(FILE_POINTS_DOT, sep=',')
        if idx >= len(df_dot):
            print(f"Error: Index {idx} out of range for dot.txt")
            return None, None, None

        beach_lon = df_dot.iloc[idx]['beach_lon']
        beach_lat = df_dot.iloc[idx]['beach_lat']
        angle_cartesian = np.nan

        return beach_lon, beach_lat, angle_cartesian
    except Exception as e:
        print(f"Error loading fixed metadata: {e}")
        return None, None, None


def main():
    print("Step 1: Loading raw data...")
    if not os.path.exists(FILE_PRED_EXCEL):
        print(f"Error: Prediction file not found: {FILE_PRED_EXCEL}")
        return

    df_pred_no = pd.read_excel(FILE_PRED_EXCEL, sheet_name=SHEET_PRED_RAW)
    df_pred_no['time'] = pd.to_datetime(df_pred_no['time'])

    df_swan_all = pd.read_csv(FILE_SWAN_DATA)
    df_swan_all['time'] = pd.to_datetime(df_swan_all['time'], format='%Y%m%d%H', errors='coerce')

    coords = load_buoy_coords()
    plot_data = []
    profile_db = {}
    target_ts = pd.Timestamp(SNAPSHOT_TIME)
    target_ts_str = target_ts.strftime('%Y%m%d%H')

    for bid, dist_m in BUOY_DISTANCES.items():
        if bid not in coords:
            continue
        obs_df = load_real_obs(bid)
        if obs_df is None:
            continue

        lat, lon = coords[bid]['lat'], coords[bid]['lon']
        target_idx = int(round(dist_m / 50.0))

        print(f"Processing Buoy: {bid}...")
        mae_era5, nearest_lon, nearest_lat, dist_to_era5_km = get_era5_mae(bid, lat, lon, obs_df)
        mae_no = get_model_mae(bid, obs_df, df_pred_no, target_idx)
        mae_swan = np.nan

        swan_df_buoy = df_swan_all[df_swan_all['id'] == bid].copy()
        if not swan_df_buoy.empty:
            swan_df_buoy = swan_df_buoy.set_index('time')
            hs_col = get_closest_column(swan_df_buoy.columns, "hs", target_idx)
            if hs_col:
                common = obs_df.index.intersection(swan_df_buoy.index)
                if len(common) > 0:
                    mae_swan = calculate_mae(obs_df.loc[common, 'hs'], swan_df_buoy.loc[common, hs_col])

        if not (np.isnan(mae_era5) or np.isnan(mae_swan) or np.isnan(mae_no)):
            b_type = 'Testing' if bid in TEST_BUOYS else 'Training'
            plot_data.append({
                'name': bid, 'type': b_type, 'dist_km': dist_m / 1000.0,
                'lon': lon, 'lat': lat,
                'mae_era5': mae_era5, 'mae_swan': mae_swan, 'mae_no': mae_no
            })

        val_obs = np.nan
        if target_ts in obs_df.index:
            val_obs = obs_df.loc[target_ts, 'hs']

        trans_dists, trans_vals = [], []
        df_p = df_pred_no[df_pred_no['id'] == bid]
        if not df_p.empty:
            row = df_p[df_p['time'] == target_ts]
            if not row.empty:
                trans_dists, trans_vals = get_profile_data(row.iloc[0], df_p.columns, "hs")

        swan_dists, swan_vals = [], []
        s_df = swan_df_buoy
        if not s_df.empty:
            if target_ts in s_df.index:
                row_s = s_df.loc[target_ts]
                swan_dists, swan_vals = get_profile_data(row_s, s_df.columns, "hs")

        boundary_val = load_boundary_value(bid, target_ts_str)

        if trans_dists or swan_dists:
            profile_db[bid] = {
                'buoy_dist_km': dist_m / 1000.0,
                'obs_val': val_obs,
                'trans_profile': (trans_dists, trans_vals),
                'swan_profile': (swan_dists, swan_vals),
                'boundary_val': boundary_val
            }
            if not np.isnan(nearest_lon):
                profile_db[bid]['nearest_era5'] = {
                    'lon': nearest_lon, 'lat': nearest_lat, 'dist_km': dist_to_era5_km
                }

    if TARGET_BUOY_DETAIL in profile_db:
        c_lon, c_lat, angle = load_fixed_coast_metadata(TARGET_BUOY_DETAIL)
        if c_lon is not None:
            profile_db[TARGET_BUOY_DETAIL]['coast_start_point'] = {
                'lon': c_lon, 'lat': c_lat, 'angle': angle
            }

    lon_min, lon_max = -6, 2
    lat_min, lat_max = 49.5, 55.5
    coast_dists, selected_lons, selected_lats = analyze_coast_to_era5_distance(
        [lon_min, lon_max, lat_min, lat_max]
    )

    data_pack = {
        'plot_data': plot_data,
        'profile_db': profile_db,
        'snapshot_time': SNAPSHOT_TIME,
        'coast_analysis': {
            'kde_dists': coast_dists,
            'plot_lons': selected_lons,
            'plot_lats': selected_lats
        },
        'extent': [lon_min, lon_max, lat_min, lat_max],
        'detail_buoy_id': TARGET_BUOY_DETAIL
    }

    with open(OUTPUT_DATA_FILE, 'wb') as f:
        pickle.dump(data_pack, f)

    print(f"\nSuccess! Processed data saved to '{OUTPUT_DATA_FILE}'.")


if __name__ == "__main__":
    main()
