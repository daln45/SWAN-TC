# -*- coding: utf-8 -*-
"""
Figure 5 – Australia (January 2021) validation summary.
Generates:
  1. Map + 4 target buoy time series
  2. R² bar chart for all Australian buoys
  3. Additional buoy time-series pages
"""

import os
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, ConnectionPatch
from pathlib import Path


# =========================
# Config
# =========================
class Config:
    data_dir      = Path(__file__).parent.parent.parent.parent / "data" / "fig4_fig5"
    buoy_file     = data_dir / "buoy_data_aus.csv"
    pred_file     = data_dir / "Aus_Final_Pred_2021_Additive_Limit.xlsx"
    metadata_path = data_dir / "offshore_points_metadata_2021_01.xlsx"
    land_shp_path = data_dir / "land_polygons.shp"

    result_dir      = Path(".")
    plot_start      = "2021-01-15 00:00:00"
    plot_end        = "2021-01-31 23:59:59"
    target_plot_buoys = ["Hay Point", "Mackay Inner", "Mooloolaba", "Tweed Heads"]
    variable        = "Hs"

    depth_indices       = list(range(10, 200, 3))[:64]
    grid_spacing        = 50
    model_grid_distances = np.array(list(range(10, 200, 3))[:64]) * 50


# Buoy distances to shore (m)
BUOY_DISTANCES = {
    "Bengello":    617.90,   "Hay Point":   1488.97,
    "Mackay Inner": 3476.57, "Mooloolaba":  8075.90,
    "Stockton":    943.00,   "Tweed Heads": 1608.34,
    "Wide Bay":    9660.66,
}

TARGET_BUOY_INFO = {
    "Hay Point":    {"lat": -21.28, "offset": (-0.28, -0.22)},
    "Mackay Inner": {"lat": -21.10, "offset": (-0.30,  0.20)},
    "Mooloolaba":   {"lat": -26.56, "offset": (-0.36,  0.00)},
    "Tweed Heads":  {"lat": -28.19, "offset": (-0.36,  0.00)},
}

LAND_COLOR      = "#CFCDCD"
OCEAN_COLOR     = "#FFFFFF"
LAND_EDGE_COLOR = "#C5C5C5"
WATERMARK_COLOR = "#555555"
GRID_COLOR      = "gray"
TARGET_EDGE_COLOR   = "black"
LEGEND_EDGE_COLOR   = "black"
TEXTBOX_BG_COLOR    = "white"
NEGATIVE_MARK_COLOR = "black"

COLORS = {
    "obs":  "#404040", "raw": "#E64B35",
    "corr": "#4DBBD5", "target": "#F39C12",
}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 14,
    "axes.linewidth": 1.5, "figure.dpi": 300,
    "savefig.dpi": 300, "mathtext.default": "regular",
})

os.makedirs(Config.result_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# =========================
# Utilities
# =========================
def calculate_r2(obs, pred):
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    valid = ~(np.isnan(obs) | np.isnan(pred))
    o, p = obs[valid], pred[valid]
    if len(o) < 2:
        return np.nan
    ss_res = np.sum((o - p) ** 2)
    ss_tot = np.sum((o - np.mean(o)) ** 2)
    return np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot


def add_panel_label(ax, label):
    ax.text(-0.02, 1.02, label, transform=ax.transAxes,
            fontsize=20, fontweight="bold", va="bottom", ha="right")


def get_best_col_index_by_distance(distance_value):
    if pd.isna(distance_value):
        return 0
    return int(np.argmin(np.abs(Config.model_grid_distances - float(distance_value))))


def prepare_distance_map(meta_df):
    dist_map = {str(k).strip(): float(v) for k, v in BUOY_DISTANCES.items()}
    if "distance_km" in meta_df.columns:
        md = (meta_df[["buoy_id", "distance_km"]]
              .dropna(subset=["buoy_id", "distance_km"])
              .drop_duplicates(subset=["buoy_id"]))
        for _, row in md.iterrows():
            bid = str(row["buoy_id"]).strip()
            if bid not in dist_map:
                dist_map[bid] = float(row["distance_km"]) * 1000.0
    return dist_map


# =========================
# Data loading
# =========================
def load_and_prepare_data():
    logger.info("Loading observation, prediction, and metadata...")
    for f in [Config.buoy_file, Config.pred_file, Config.metadata_path]:
        if not Path(f).exists():
            raise FileNotFoundError(f"File not found: {f}")

    obs_df = pd.read_csv(Config.buoy_file)
    obs_df = obs_df.rename(columns={
        "Buoy_ID": "id", "DateTimeGMT": "time",
        "Hs_m": "hs_obs", "Tm02_s": "tm_obs",
        "hs": "hs_obs", "Tm": "tm_obs",
    })
    obs_df["time"] = pd.to_datetime(obs_df["time"].astype(str), format="%Y%m%d%H", errors="coerce")
    obs_df["time"] = obs_df["time"].dt.tz_localize(None)
    obs_df = obs_df.dropna(subset=["time", "id"])
    obs_df["id"] = obs_df["id"].astype(str).str.strip()

    meta_df = pd.read_excel(Config.metadata_path)
    meta_df["buoy_id"] = meta_df["buoy_id"].astype(str).str.strip()
    dist_map = prepare_distance_map(meta_df)

    pred_meta = pd.read_excel(Config.pred_file, sheet_name="Meta", engine="openpyxl")
    pred_meta["time"] = pd.to_datetime(pred_meta["time"], errors="coerce").dt.tz_localize(None)
    pred_meta["id"]   = pred_meta["id"].astype(str).str.strip()

    raw_hs   = pd.read_excel(Config.pred_file, sheet_name="Hs_Raw",       engine="openpyxl")
    corr_hs  = pd.read_excel(Config.pred_file, sheet_name="Hs_Corrected", engine="openpyxl")
    raw_tm   = pd.read_excel(Config.pred_file, sheet_name="Tm_Raw",       engine="openpyxl")
    corr_tm  = pd.read_excel(Config.pred_file, sheet_name="Tm_Corrected", engine="openpyxl")

    for df in [raw_hs, corr_hs, raw_tm, corr_tm]:
        df.drop(columns=["id", "time", "Buoy_ID", "Time"], inplace=True, errors="ignore")

    arr_hs_raw   = raw_hs.values
    arr_hs_corr  = corr_hs.values
    arr_tm_raw   = raw_tm.values
    arr_tm_corr  = corr_tm.values

    pred_meta["distance_value"] = pred_meta["id"].map(dist_map)
    col_idx  = pred_meta["distance_value"].apply(get_best_col_index_by_distance).values.astype(int)
    row_idx  = np.arange(len(pred_meta))

    pred_meta["hs_pred_raw"]  = arr_hs_raw [row_idx, col_idx]
    pred_meta["hs_pred_corr"] = arr_hs_corr[row_idx, col_idx]
    pred_meta["tm_pred_raw"]  = arr_tm_raw [row_idx, col_idx]
    pred_meta["tm_pred_corr"] = arr_tm_corr[row_idx, col_idx]

    merged = pd.merge(pred_meta, obs_df, on=["id", "time"], how="inner")
    merged = merged[
        (merged["time"] >= pd.Timestamp(Config.plot_start)) &
        (merged["time"] <= pd.Timestamp(Config.plot_end))
    ]
    logger.info(f"Merged observations: {len(merged)}")
    return merged, meta_df


# =========================
# Figure 1: Map + time series
# =========================
def draw_map_plus_timeseries(merged_df, buoy_meta_df, variable="Hs"):
    logger.info("Drawing Figure 1: map + 4 buoy time series...")
    col_obs, col_raw, col_corr = (
        ("hs_obs", "hs_pred_raw", "hs_pred_corr") if variable == "Hs"
        else ("tm_obs", "tm_pred_raw", "tm_pred_corr")
    )
    ylabel   = "Significant Wave Height (m)" if variable == "Hs" else "Mean Wave Period (s)"
    out_name = f"Figure5_Map_TimeSeries_{variable}_Jan2021.png"

    buoy_meta_df = buoy_meta_df.copy()
    buoy_meta_df["buoy_id"] = buoy_meta_df["buoy_id"].astype(str).str.strip()
    buoy_meta_df = buoy_meta_df[
        buoy_meta_df["buoy_id"].isin(set(BUOY_DISTANCES.keys()))
    ].reset_index(drop=True)

    if buoy_meta_df.empty:
        logger.warning("No buoy metadata available, skipping Figure 1.")
        return

    valid_targets = [b for b in Config.target_plot_buoys if b in buoy_meta_df["buoy_id"].values]
    if not valid_targets:
        logger.warning("No target buoys found in metadata.")
        return

    lat_map       = buoy_meta_df.set_index("buoy_id")["beach_lat"].to_dict()
    sorted_targets = sorted(
        valid_targets,
        key=lambda x: TARGET_BUOY_INFO.get(x, {}).get("lat", lat_map.get(x, -999)),
        reverse=True,
    )[:4]

    lon_min, lon_max = 144.5, 155.0
    lat_min, lat_max = -38.0, -16.0

    fig = plt.figure(figsize=(19, 14))
    gs  = gridspec.GridSpec(4, 2, width_ratios=[0.7, 1], height_ratios=[1, 1, 1, 1],
                            left=0.05, right=0.98, bottom=0.05, top=0.95,
                            wspace=0.05, hspace=0.25)

    ax_map = fig.add_subplot(gs[:, 0])
    ax_map.set_facecolor(OCEAN_COLOR)

    if Path(str(Config.land_shp_path)).exists():
        try:
            land = gpd.read_file(Config.land_shp_path)
            if land.crs and land.crs.to_epsg() != 4326:
                land = land.to_crs("EPSG:4326")
            land.cx[lon_min:lon_max, lat_min:lat_max].plot(
                ax=ax_map, color=LAND_COLOR, edgecolor=LAND_EDGE_COLOR, linewidth=0.8, zorder=1)
            ax_map.text(147.5, -32.0, "Australia", fontsize=36, fontweight="bold",
                        fontstyle="italic", color=WATERMARK_COLOR, alpha=0.8,
                        ha="center", va="center", zorder=2)
        except Exception as e:
            logger.warning(f"Failed to load land shapefile: {e}")

    map_points, ts_axes = {}, {}

    for idx, bid in enumerate(buoy_meta_df.sort_values("beach_lat", ascending=False)["buoy_id"].tolist()):
        row = buoy_meta_df[buoy_meta_df["buoy_id"] == bid].iloc[0]
        x, y = row["beach_lon"], row["beach_lat"]
        if bid in sorted_targets:
            map_points[bid] = (x, y)
        ax_map.scatter(x, y, c=COLORS["target"], s=180, edgecolors=TARGET_EDGE_COLOR,
                       linewidths=1.5, zorder=6, label="Buoys" if idx == 0 else None)
        off_x, off_y = TARGET_BUOY_INFO.get(bid, {}).get("offset", (-0.30, 0.00))
        ax_map.text(x + off_x, y + off_y, bid, fontsize=16, fontweight="bold",
                    ha="right", va="center", zorder=7)

    ax_map.set_xlim(lon_min, lon_max)
    ax_map.set_ylim(lat_min, lat_max)
    ax_map.grid(True, linestyle="--", alpha=0.35, color=GRID_COLOR)
    ax_map.set_xlabel("Longitude", fontsize=22, fontweight="bold", labelpad=10)
    ax_map.set_ylabel("Latitude",  fontsize=22, fontweight="bold", labelpad=10)
    ax_map.set_xticks(np.arange(145, 156, 4))
    ax_map.set_yticks(np.arange(-37, -15, 3))
    ax_map.set_xticklabels([f"{x}°E" for x in np.arange(145, 156, 4)], fontsize=16, fontweight="bold")
    ax_map.set_yticklabels([f"{abs(y)}°S" for y in np.arange(-37, -15, 3)], fontsize=16, fontweight="bold")
    ax_map.tick_params(axis="both", labelsize=16, width=1.5)
    ax_map.legend(loc="lower left", fontsize=14, frameon=True, edgecolor=LEGEND_EDGE_COLOR)
    add_panel_label(ax_map, "a")

    panel_labels = ["b", "c", "d", "e"]
    for i, bid in enumerate(sorted_targets):
        ax_ts = fig.add_subplot(gs[i, 1])
        ts_axes[bid] = ax_ts

        df_sub = merged_df[merged_df["id"] == bid].sort_values("time")
        if df_sub.empty:
            ax_ts.text(0.5, 0.5, "No Data", ha="center", fontsize=16,
                       fontweight="bold", transform=ax_ts.transAxes)
            continue

        times = df_sub["time"]
        obs   = df_sub[col_obs].values
        raw   = df_sub[col_raw].values
        corr  = df_sub[col_corr].values

        ax_ts.scatter(times, obs,  color=COLORS["obs"],  s=18, alpha=0.7, marker="o", label="Observed", zorder=5)
        ax_ts.plot(times,   raw,   color=COLORS["raw"],  lw=2.2, label="SWAN-T",  zorder=3)
        ax_ts.plot(times,   corr,  color=COLORS["corr"], lw=2.2, label="SWAN-TC", zorder=4)

        r2_raw  = calculate_r2(obs, raw)
        r2_corr = calculate_r2(obs, corr)
        ax_ts.text(0.98, 0.05, f"SWAN-T: R²={r2_raw:.2f}\nSWAN-TC: R²={r2_corr:.2f}",
                   transform=ax_ts.transAxes, ha="right", va="bottom",
                   fontsize=16, fontweight="bold",
                   bbox=dict(facecolor=TEXTBOX_BG_COLOR, alpha=0.8, edgecolor="none"))

        dist_m    = BUOY_DISTANCES.get(bid, np.nan)
        dist_text = f"{float(dist_m) / 1000.0:.2f} km" if pd.notna(dist_m) else "NA"
        ax_ts.set_title(f"{bid} (distance to shore: {dist_text})",
                        loc="left", fontsize=18, fontweight="bold", pad=8)
        add_panel_label(ax_ts, panel_labels[i])
        ax_ts.set_xlim(pd.Timestamp(Config.plot_start), pd.Timestamp(Config.plot_end))
        ax_ts.set_ylabel(ylabel, fontsize=16, fontweight="bold")
        ax_ts.grid(True, linestyle="--", alpha=0.35)
        ax_ts.tick_params(axis="y", labelsize=14, width=1.5)

        if i == len(sorted_targets) - 1:
            ax_ts.set_xlabel("Date (2021-01)", fontsize=18, fontweight="bold", labelpad=10)
            ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            ax_ts.tick_params(axis="x", labelsize=14, rotation=0, width=1.5)
        else:
            ax_ts.set_xticklabels([])

        if i == 0:
            ax_ts.legend(loc="upper left", ncol=1, fontsize=14, frameon=True,
                         edgecolor=LEGEND_EDGE_COLOR)

    for bid in sorted_targets:
        if bid in map_points and bid in ts_axes:
            con = ConnectionPatch(
                xyA=map_points[bid], xyB=(0, 0.5),
                coordsA="data", coordsB="axes fraction",
                axesA=ax_map, axesB=ts_axes[bid],
                color="#555555", lw=1.5, linestyle="--", alpha=0.6, zorder=20,
            )
            fig.add_artist(con)

    plt.savefig(Config.result_dir / out_name, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Figure 1 saved: {Config.result_dir / out_name}")


# =========================
# Figure 2: R² bar chart
# =========================
def draw_r2_bar_all_buoys(merged_df, buoy_meta_df, variable="Hs"):
    logger.info("Drawing Figure 2: R² bar chart for all buoys...")
    col_obs, col_raw, col_corr = (
        ("hs_obs", "hs_pred_raw", "hs_pred_corr") if variable == "Hs"
        else ("tm_obs", "tm_pred_raw", "tm_pred_corr")
    )
    ylab     = "R² (Significant Wave Height)" if variable == "Hs" else "R² (Mean Wave Period)"
    out_name = f"Figure5_R2_AllBuoys_{variable}_Jan2021.png"

    buoy_meta_df = buoy_meta_df.copy()
    buoy_meta_df["buoy_id"] = buoy_meta_df["buoy_id"].astype(str).str.strip()
    buoy_meta_df = buoy_meta_df[
        buoy_meta_df["buoy_id"].isin(set(BUOY_DISTANCES.keys()))
    ].reset_index(drop=True)

    if buoy_meta_df.empty:
        logger.warning("No buoy metadata, skipping R² bar chart.")
        return

    order = buoy_meta_df.sort_values("beach_lat", ascending=False)["buoy_id"].tolist()
    rows  = []
    for bid in order:
        sub = merged_df[merged_df["id"] == bid]
        rows.append({
            "id":       bid,
            "r2_raw":   calculate_r2(sub[col_obs].values, sub[col_raw].values)  if not sub.empty else np.nan,
            "r2_corr":  calculate_r2(sub[col_obs].values, sub[col_corr].values) if not sub.empty else np.nan,
            "count":    len(sub),
        })

    r2_df = pd.DataFrame(rows)
    dist_km_map = {str(k).strip(): float(v) / 1000.0 for k, v in BUOY_DISTANCES.items()}

    x     = np.arange(len(r2_df))
    width = 0.36

    raw_vals  = r2_df["r2_raw"].values
    corr_vals = r2_df["r2_corr"].values

    fig, ax = plt.subplots(figsize=(20, 7))
    ax.bar(x - width / 2, np.where(np.isnan(raw_vals),  np.nan, np.maximum(raw_vals,  0.0)),
           width=width, color=COLORS["raw"],  label="SWAN-T")
    ax.bar(x + width / 2, np.where(np.isnan(corr_vals), np.nan, np.maximum(corr_vals, 0.0)),
           width=width, color=COLORS["corr"], label="SWAN-TC")

    ax.set_ylabel(ylab, fontsize=18, fontweight="bold")
    ax.set_xlabel("Buoy (distance to shore)", fontsize=18, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{bid}\n({dist_km_map.get(str(bid).strip(), np.nan):.2f} km)"
         if pd.notna(dist_km_map.get(str(bid).strip(), np.nan)) else f"{bid}\n(NA)"
         for bid in r2_df["id"].values],
        rotation=0, ha="center", fontsize=12, fontweight="bold"
    )
    ax.tick_params(axis="x", labelsize=12, width=1.5, pad=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="y", labelsize=14, width=1.5)
    ax.legend(loc="upper left", fontsize=14, frameon=True, edgecolor=LEGEND_EDGE_COLOR)

    mean_raw  = np.nanmean(raw_vals)
    mean_corr = np.nanmean(corr_vals)
    ax.text(0.98, 0.98, f"Mean R²\nSWAN-T: {mean_raw:.2f}\nSWAN-TC: {mean_corr:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=15, fontweight="bold")

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(Config.result_dir / out_name, dpi=300)
    plt.close()
    logger.info(f"Figure 2 saved: {Config.result_dir / out_name}")
    r2_df.to_csv(Config.result_dir / f"R2_Table_{variable}_Jan2021.csv", index=False, float_format="%.4f")


# =========================
# Main
# =========================
def main():
    try:
        merged_df, buoy_meta_df = load_and_prepare_data()
        if merged_df.empty:
            logger.error("Merged data empty, cannot plot.")
            return
        draw_map_plus_timeseries(merged_df, buoy_meta_df, variable=Config.variable)
        draw_r2_bar_all_buoys(merged_df, buoy_meta_df,   variable=Config.variable)
        logger.info("All figures complete.")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
