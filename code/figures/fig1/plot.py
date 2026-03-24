# -*- coding: utf-8 -*-
"""
Figure 1 – UK buoy map, wave profile comparison, and MAE grouped bar chart.
Loads processed data from prepare_data.py output (processed_plot_data_3x3.pkl).
"""

import os
import math
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

warnings.filterwarnings('ignore')

# =========================
# Config
# =========================
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "fig1"
INPUT_DATA_FILE = DATA_DIR / "processed_plot_data_3x3.pkl"
SHAPEFILE_PATH  = DATA_DIR / "land_polygons.shp"

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'mathtext.default': 'regular',
})

STYLES = {
    'obs':   {'color': '#2C3E50', 'lw': 2.0, 'ls': '-',  'marker': 'o', 'ms': 5, 'label': 'Observation'},
    'swan':  {'color': '#E74C3C', 'lw': 1.8, 'ls': '--', 'marker': None, 'ms': 0, 'label': 'SWAN'},
    'trans': {'color': '#3498DB', 'lw': 1.8, 'ls': '-',  'marker': None, 'ms': 0, 'label': 'SWAN-TC'},
    'era5':  {'color': '#27AE60', 'lw': 1.5, 'ls': ':',  'marker': None, 'ms': 0, 'label': 'ERA5'},
    'bound': {'color': '#8E44AD', 'lw': 1.5, 'ls': '-.',  'marker': 's', 'ms': 6, 'label': 'Boundary'},
}

NAME_MAPPING = {
    'WBy_waves': 'West Bay',    'PBy_waves': 'Pevensey Bay',
    'Fxs_waves': 'Felixstowe', 'Hrn_waves': 'Hornsea',
}


# =========================
# Helper functions
# =========================
def calculate_destination_point(lon, lat, bearing_deg, distance_km):
    R = 6371.0
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    bearing = math.radians(bearing_deg)
    d_over_R = distance_km / R
    lat2 = math.asin(
        math.sin(lat1) * math.cos(d_over_R) +
        math.cos(lat1) * math.sin(d_over_R) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(d_over_R) * math.cos(lat1),
        math.cos(d_over_R) - math.sin(lat1) * math.sin(lat2)
    )
    return math.degrees(lon2), math.degrees(lat2)


def setup_map_axis(ax, extent):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#EFEFEF', edgecolor='#888888', linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='#D6EAF8')
    ax.add_feature(cfeature.COASTLINE, edgecolor='#555555', linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, edgecolor='#AAAAAA', linewidth=0.4)
    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False,
                      linewidth=0.4, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False


def plot_land(ax, shp_path):
    import geopandas as gpd
    if not os.path.exists(shp_path):
        return
    gdf = gpd.read_file(shp_path)
    gdf.plot(ax=ax, color='#EFEFEF', edgecolor='#888888', linewidth=0.5,
             transform=ccrs.PlateCarree())


def plot_grouped_barplot(ax, plot_data, test_buoys):
    test_data  = [d for d in plot_data if d['type'] == 'Testing']
    train_data = [d for d in plot_data if d['type'] == 'Training']

    categories = ['ERA5', 'SWAN', 'SWAN-TC']
    x = np.arange(len(categories))
    width = 0.35

    colors_train = ['#2ECC71', '#E74C3C', '#3498DB']
    colors_test  = ['#1A8742', '#C0392B', '#1A6FA8']

    for i, d in enumerate(test_data):
        vals = [d['mae_era5'], d['mae_swan'], d['mae_no']]
        name = NAME_MAPPING.get(d['name'], d['name'])
        ax.bar(x + i * width / max(len(test_data), 1), vals, width / max(len(test_data), 1),
               color=colors_test, alpha=0.8, label=f'{name} (Test)')

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(categories, fontweight='bold')
    ax.set_ylabel('MAE (m)', fontweight='bold')
    ax.set_title('Model MAE Comparison', fontweight='bold')
    ax.legend(fontsize=9, frameon=True)
    ax.grid(True, axis='y', alpha=0.3, ls='--')


def main():
    if not os.path.exists(INPUT_DATA_FILE):
        print(f"Error: data file not found: {INPUT_DATA_FILE}")
        print("Please run prepare_data.py first to generate the processed data.")
        return

    with open(INPUT_DATA_FILE, 'rb') as f:
        data_pack = pickle.load(f)

    plot_data   = data_pack['plot_data']
    profile_db  = data_pack['profile_db']
    extent      = data_pack['extent']
    detail_id   = data_pack.get('detail_buoy_id', 'WBy_waves')

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # Panel a: UK map
    ax_map = fig.add_subplot(gs[:, 0], projection=ccrs.PlateCarree())
    setup_map_axis(ax_map, [extent[0] - 0.5, extent[1] + 0.5, extent[2] - 0.5, extent[3] + 0.5])

    test_buoys = [d['name'] for d in plot_data if d['type'] == 'Testing']
    for d in plot_data:
        color = '#E74C3C' if d['type'] == 'Testing' else '#3498DB'
        ax_map.scatter(d['lon'], d['lat'], c=color, s=40, zorder=5,
                       transform=ccrs.PlateCarree())
        name = NAME_MAPPING.get(d['name'], d['name'])
        if d['type'] == 'Testing':
            ax_map.text(d['lon'] + 0.05, d['lat'], name, fontsize=8, fontweight='bold',
                        transform=ccrs.PlateCarree())

    ax_map.set_title('a UK wave buoy network', loc='left', fontweight='bold', fontsize=14)
    train_patch = mpatches.Patch(color='#3498DB', label='Training buoys')
    test_patch  = mpatches.Patch(color='#E74C3C', label='Test buoys')
    ax_map.legend(handles=[train_patch, test_patch], loc='lower left', fontsize=9)

    # Panel b: Detail map for target buoy
    ax_detail = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    if detail_id in profile_db:
        d_info = profile_db[detail_id]
        lon_c  = plot_data[[d for d in plot_data if d['name'] == detail_id][0]['lon'] if
                            any(d['name'] == detail_id for d in plot_data) else 0]
        lat_c  = plot_data[[d for d in plot_data if d['name'] == detail_id][0]['lat'] if
                            any(d['name'] == detail_id for d in plot_data) else 0]
    ax_detail.set_title('b Profile transect location', loc='left', fontweight='bold', fontsize=14)

    # Panel c: Wave profile comparison
    ax_profile = fig.add_subplot(gs[1, 1])
    if detail_id in profile_db:
        pdb = profile_db[detail_id]
        td, tv = pdb['trans_profile']
        sd, sv = pdb['swan_profile']
        if td:
            ax_profile.plot(td, tv, **{k: v for k, v in STYLES['trans'].items() if k != 'marker'})
        if sd:
            ax_profile.plot(sd, sv, **{k: v for k, v in STYLES['swan'].items() if k != 'marker'})
        if not np.isnan(pdb['obs_val']):
            ax_profile.axhline(pdb['obs_val'], color=STYLES['obs']['color'], ls='--', lw=1.5,
                               label=f"Obs: {pdb['obs_val']:.2f} m")
        ax_profile.set_xlabel('Distance to shore (km)', fontweight='bold')
        ax_profile.set_ylabel('SWH (m)', fontweight='bold')
        ax_profile.set_title('c Wave height profile', loc='left', fontweight='bold', fontsize=14)
        ax_profile.legend(fontsize=9)
        ax_profile.grid(True, alpha=0.3, ls='--')

    # Panel d: MAE bar chart
    ax_bar = fig.add_subplot(gs[:, 2])
    if plot_data:
        plot_grouped_barplot(ax_bar, plot_data, test_buoys)

    plt.savefig('Figure1_UK_Overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 1 saved -> Figure1_UK_Overview.png")


if __name__ == "__main__":
    main()
