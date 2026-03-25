# -*- coding: utf-8 -*-
"""
Figure 2 – Spatial R² distribution plots for the four test buoys.
Compares SWAN-T trained with and without artificial beach profiles.
"""

import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

# =========================
# Config
# =========================
BUOY_CONFIG = {
    'Fxs_waves': {'distance': 3308.10, 'name': 'Fxs'},
    'Hrn_waves': {'distance': 7015.19, 'name': 'Hrn'},
    'PBy_waves': {'distance': 5294.04, 'name': 'PBy'},
    'WBy_waves': {'distance': 1198.36, 'name': 'WBy'},
}

NAME_MAPPING = {
    'WBy': 'West Bay',
    'PBy': 'Pevensey Bay',
    'Fxs': 'Felixstowe',
    'Hrn': 'Hornsea',
}

DEPTH_INDICES = list(range(10, 200, 3))
GRID_SPACING = 50

# Paths relative to this script file
_BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent   # repo root
_DATA_DIR = _BASE_DIR / "data"
_RESULTS_DIR = _BASE_DIR / "results"

DIR_BOT_FILES = str(_DATA_DIR / "fig2" / "bot")

RESULT_DIR_BASE = _RESULTS_DIR / "swan_t"
FILE_SWAN_HS = _DATA_DIR / "uk" / "swan_hs_2020.csv"
FILE_SWAN_TM = _DATA_DIR / "uk" / "swan_tm_2020.csv"

EXP_NAME_SYNTHETIC  = "exp_5_100_percent"
EXP_NAME_REAL_ONLY  = "exp_7_100_real_only"

# =========================
# Plot style
# =========================
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['mathtext.default'] = 'regular'

COLOR_OBSERVATION       = '#2C3E50'
COLOR_NO_SYNTHETIC      = '#E64B35'
COLOR_WITH_SYNTHETIC    = '#4DBBD5'
COLOR_WATER             = '#ACE2FF'
COLOR_TERRAIN           = '#EAD9B0'
COLOR_LINE_NO           = '#7A7A7A'
COLOR_LINE_SY           = '#000000'


# =========================
# Data loading
# =========================
def map_columns_index_to_depth(df):
    rename_dict = {}
    pattern = re.compile(r'^pred_(hs|tm)_(\d+)$')
    for col in df.columns:
        match = pattern.match(col)
        if match:
            var_type = match.group(1)
            idx_seq = int(match.group(2))
            depth_idx = 10 + idx_seq * 3
            rename_dict[col] = f"pred_{var_type}{depth_idx:03d}"
    if rename_dict:
        df.rename(columns=rename_dict, inplace=True)
    return df


def load_transformer_data(exp_name):
    result_file = RESULT_DIR_BASE / f"results_data_efficiency_{exp_name}" / "train_test_results" / f"results_{exp_name}.xlsx"
    if not result_file.exists():
        print(f"Result file not found: {result_file}")
        return None
    try:
        df = pd.read_excel(result_file, engine='openpyxl')
        df.rename(columns={'id': 'Buoy_ID', 'time': 'Time'}, inplace=True)
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'], format='%Y%m%d%H', errors='coerce')
        df = map_columns_index_to_depth(df)
        return df
    except Exception as e:
        print(f"Error loading {exp_name}: {e}")
        return None


def load_swan_data(file_path):
    if not Path(file_path).exists():
        return None
    try:
        df = pd.read_csv(file_path)
        if 'time' in df.columns:
            df['Time'] = pd.to_datetime(df['time'], format='%Y%m%d%H', errors='coerce')
        if 'id' in df.columns:
            df.rename(columns={'id': 'Buoy_ID'}, inplace=True)
        return df
    except Exception:
        return None


def load_bathymetry(buoy_id):
    bot_dir = Path(DIR_BOT_FILES)
    if not bot_dir.exists():
        return None, None
    station_name = buoy_id.split('_')[0]
    candidates = [
        f"{buoy_id}.bot", f"{station_name}.bot",
        f"{station_name.lower()}.bot", f"{station_name.upper()}.bot"
    ]
    for fname in candidates:
        full_path = bot_dir / fname
        if full_path.exists():
            data = np.loadtxt(full_path)
            return np.linspace(0, 10, len(data)), data
    return None, None


# =========================
# Plotting
# =========================
def plot_single_spatial_subplot(ax, buoy_id, df_no, df_sy, df_true, var_type, letter, highlight_dist=None):
    info = BUOY_CONFIG[buoy_id]
    buoy_name = NAME_MAPPING.get(info['name'], info['name'])
    x_km_all = np.array(DEPTH_INDICES) * GRID_SPACING / 1000.0

    d_bathy, z_bathy = load_bathymetry(buoy_id)
    ax_bathy = None
    if d_bathy is not None:
        ax_bathy = ax.twinx()
        ax_bathy.yaxis.set_major_locator(MaxNLocator(nbins=4))
        mask_b = d_bathy <= 10.2
        db, zb = d_bathy[mask_b], z_bathy[mask_b]
        if len(db) > 0:
            max_depth = zb.max() * 1.15 if zb.max() > 0 else 50
            ax_bathy.fill_between(db, 0, zb, color=COLOR_WATER, alpha=1, zorder=0)
            ax_bathy.fill_between(db, zb, max_depth, color=COLOR_TERRAIN, alpha=1, zorder=0)
            ax_bathy.plot(db, zb, color='black', linewidth=1.8, alpha=0.6, zorder=2)

            seabed_style = dict(color='#590003', fontsize=14, fontweight='bold',
                                ha='center', va='top', zorder=5, alpha=0.9)
            if db.min() < 3 < db.max():
                ax_bathy.text(
                    db[np.argmin(np.abs(db - 3))],
                    z_bathy[np.argmin(np.abs(db - 3))] + max_depth * 0.065,
                    "Seabed", **seabed_style
                )
            if db.min() < 9 < db.max():
                ax_bathy.text(
                    db[np.argmin(np.abs(db - 9))],
                    z_bathy[np.argmin(np.abs(db - 9))] + max_depth * 0.025,
                    "Seabed", **seabed_style
                )

            ax_bathy.set_ylim(max_depth, 0)
            ax_bathy.tick_params(axis='y', labelcolor='#5998CE')
            ax_bathy.set_ylabel('Water depth (m)', color='#5998CE', fontweight='bold')

    r2_no_list, r2_sy_list = [], []
    df_no_c   = df_no[df_no['Buoy_ID'] == buoy_id]
    df_sy_c   = df_sy[df_sy['Buoy_ID'] == buoy_id]
    df_true_c = df_true[df_true['Buoy_ID'] == buoy_id]

    def calc(dp, dt, cp, ct):
        m = pd.merge(dp[['Time', cp]], dt[['Time', ct]], on='Time').dropna()
        return r2_score(m[ct], m[cp]) if len(m) > 10 else np.nan

    for idx in DEPTH_INDICES:
        cp = f'pred_{var_type}{idx:03d}'
        ct = f'{var_type}{idx:03d}'
        r2_no_list.append(calc(df_no_c, df_true_c, cp, ct))
        r2_sy_list.append(calc(df_sy_c, df_true_c, cp, ct))

    r2_no = np.array(r2_no_list)
    r2_sy = np.array(r2_sy_list)

    mask = (x_km_all >= 1.0) & (x_km_all <= 10.0)
    x_plot = x_km_all[mask]
    if np.any(mask):
        ax.plot(x_plot[::2], r2_no[mask][::2], color=COLOR_LINE_NO, ls='-', marker='o',
                ms=6, lw=2, label='SWAN-T', zorder=10)
        ax.plot(x_plot[::2], r2_sy[mask][::2], color=COLOR_LINE_SY, ls='-', marker='o',
                ms=6, lw=2, label='SWAN-T (trained with artificial profiles)', zorder=10)

    if highlight_dist is not None:
        idx_candidates = np.where(np.isclose(x_km_all, highlight_dist))[0]
        if len(idx_candidates) > 0:
            idx_h = idx_candidates[0]
            ax.scatter([highlight_dist], [r2_no[idx_h]], color='#FF0000', marker='o',
                       s=100, zorder=20, edgecolors='black', linewidth=1.5)
            ax.scatter([highlight_dist], [r2_sy[idx_h]], color='#FF0000', marker='o',
                       s=100, zorder=20, edgecolors='black', linewidth=1.5)

    ax.axhline(0.9, color='gray', ls='--', alpha=0.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_ylim(0.60 if var_type == 'hs' else 0.20, 1.02)
    ax.set_xlim(1.0, 10.0)
    ax.grid(True, alpha=0.3, ls='--')
    ax.set_title(f'{letter} {buoy_name}', loc='left', fontweight='bold', fontsize=16)
    ax.set_xlabel('Distance to shore (km)', fontweight='bold')
    ax.set_ylabel('R²', fontweight='bold')

    if ax_bathy:
        ax.set_zorder(10)
        ax_bathy.set_zorder(1)
        ax.patch.set_visible(False)

    return r2_no, r2_sy


def plot_hs_figure_1(df_no, df_sy, df_true):
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=0.25, wspace=0.25)

    ax_a = fig.add_subplot(gs[0, 0])
    plot_single_spatial_subplot(ax_a, 'WBy_waves', df_no, df_sy, df_true, 'hs', 'a')

    r2_no_fxs, r2_sy_fxs = plot_single_spatial_subplot(
        fig.add_subplot(gs[0, 1]), 'Fxs_waves', df_no, df_sy, df_true, 'hs', 'b'
    )
    plt.delaxes(fig.axes[-1])

    x_km = np.array(DEPTH_INDICES) * GRID_SPACING / 1000.0
    mask_range = (x_km >= 1.0) & (x_km <= 10.0)
    valid_indices = np.where(mask_range)[0]
    diff = r2_sy_fxs - r2_no_fxs
    best_local_idx = np.nanargmax(diff[valid_indices])
    best_global_idx = valid_indices[best_local_idx]
    best_depth_idx = DEPTH_INDICES[best_global_idx]
    best_dist_km = x_km[best_global_idx]

    ax_b = fig.add_subplot(gs[0, 1])
    plot_single_spatial_subplot(ax_b, 'Fxs_waves', df_no, df_sy, df_true, 'hs', 'b',
                                highlight_dist=best_dist_km)
    ax_b.legend(loc='lower left', frameon=False, fontsize=11)

    col_p = f'pred_hs{best_depth_idx:03d}'
    col_t = f'hs{best_depth_idx:03d}'
    df_n_f = df_no[df_no['Buoy_ID'] == 'Fxs_waves'][['Time', col_p]].rename(columns={col_p: 'no'})
    df_s_f = df_sy[df_sy['Buoy_ID'] == 'Fxs_waves'][['Time', col_p]].rename(columns={col_p: 'sy'})
    df_t_f = df_true[df_true['Buoy_ID'] == 'Fxs_waves'][['Time', col_t]].rename(columns={col_t: 'true'})
    m = df_t_f.merge(df_n_f, on='Time').merge(df_s_f, on='Time').dropna()
    m = m[m['Time'].dt.month.isin([9, 10])].sort_values('Time')

    ax_c = fig.add_subplot(gs[1, :])
    ax_c.scatter(m['Time'], m['true'], color=COLOR_OBSERVATION, s=15, alpha=0.7, label='SWAN', zorder=3)
    ax_c.plot(m['Time'], m['no'], color=COLOR_NO_SYNTHETIC, lw=2.0, alpha=0.8, label='SWAN-T', zorder=2)
    ax_c.plot(m['Time'], m['sy'], color=COLOR_WITH_SYNTHETIC, lw=2.0, alpha=0.8,
              label='SWAN-T (trained with artificial profiles)', zorder=4)

    ax_c.set_title('c Time series at Felixstowe', loc='left', fontweight='bold', fontsize=16)
    ax_c.xaxis.set_major_locator(mdates.DayLocator(interval=10))
    ax_c.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax_c.set_xlabel('Time (2020)', fontweight='bold')
    ax_c.set_ylabel('Significant Wave Height (SWH, m)', fontweight='bold')
    ax_c.grid(True, alpha=0.3, ls='--')
    ax_c.legend(loc='upper left', frameon=True, fontsize=12, ncol=1)

    plt.savefig('Final_Fig_HS_Part1_WBy_Fxs.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_hs_figure_2(df_no, df_sy, df_true):
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.20, wspace=0.25)

    ax_a = fig.add_subplot(gs[0, 0])
    plot_single_spatial_subplot(ax_a, 'PBy_waves', df_no, df_sy, df_true, 'hs', 'a')
    ax_b = fig.add_subplot(gs[0, 1])
    plot_single_spatial_subplot(ax_b, 'Hrn_waves', df_no, df_sy, df_true, 'hs', 'b')
    ax_b.legend(loc='lower left', frameon=False, fontsize=11)

    buoys = [('PBy_waves', gs[1, 0], 'c'), ('Hrn_waves', gs[1, 1], 'd')]
    for b_id, pos, letter in buoys:
        ax = fig.add_subplot(pos)
        dist_m = BUOY_CONFIG[b_id]['distance']
        dists_all = np.array(DEPTH_INDICES) * GRID_SPACING
        idx_pos = np.argmin(np.abs(dists_all - dist_m))
        target_idx = DEPTH_INDICES[idx_pos]

        col_p = f'pred_hs{target_idx:03d}'
        col_t = f'hs{target_idx:03d}'
        sub_t = df_true[df_true['Buoy_ID'] == b_id][['Time', col_t]].rename(columns={col_t: 'true'})
        sub_n = df_no[df_no['Buoy_ID'] == b_id][['Time', col_p]].rename(columns={col_p: 'no'})
        sub_s = df_sy[df_sy['Buoy_ID'] == b_id][['Time', col_p]].rename(columns={col_p: 'sy'})

        m = sub_t.merge(sub_n, on='Time').merge(sub_s, on='Time')
        m = m[m['Time'].dt.month.isin([1])].sort_values('Time')

        ax.scatter(m['Time'], m['true'], color=COLOR_OBSERVATION, s=15, alpha=0.7, label='SWAN', zorder=2)
        ax.plot(m['Time'], m['no'], color=COLOR_NO_SYNTHETIC, lw=2.0, alpha=0.8, label='SWAN-T', zorder=2)
        ax.plot(m['Time'], m['sy'], color=COLOR_WITH_SYNTHETIC, lw=2.0, alpha=0.8,
                label='SWAN-T (trained with artificial profiles)', zorder=2)

        buoy_name = NAME_MAPPING[BUOY_CONFIG[b_id]['name']]
        ax.set_title(f'{letter} {buoy_name} ({dist_m / 1000:.2f} km till shore)',
                     loc='left', fontweight='bold', fontsize=16)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        ax.set_xlabel('Time (2020)', fontweight='bold')
        ax.set_ylabel('Significant Wave Height (SWH, m)', fontweight='bold')
        ax.grid(True, alpha=0.3, ls='--')
        if letter == 'c':
            ax.legend(loc='upper left', fontsize=10, frameon=True)

    plt.savefig('Final_Fig_HS_Part2_PBy_Hrn.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_tm_spatial_all(df_no, df_sy, df_true):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.25, wspace=0.25)
    buoys = [('WBy_waves', 'a'), ('Fxs_waves', 'b'), ('PBy_waves', 'c'), ('Hrn_waves', 'd')]
    for i, (b_id, letter) in enumerate(buoys):
        ax = axes[i // 2, i % 2]
        plot_single_spatial_subplot(ax, b_id, df_no, df_sy, df_true, 'tm', letter)
        ax.tick_params(axis='both', which='major', labelsize=14)
        if letter == 'b':
            ax.legend(loc='lower left', frameon=False, fontsize=11)
    plt.savefig('Final_Fig_TM_Spatial_All.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_tm_temporal_all(df_no, df_sy, df_true):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    buoys = [('WBy_waves', 'a'), ('Fxs_waves', 'b'), ('PBy_waves', 'c'), ('Hrn_waves', 'd')]
    dists_m = np.array(DEPTH_INDICES) * GRID_SPACING

    for i, (b_id, letter) in enumerate(buoys):
        ax = axes[i // 2, i % 2]
        info = BUOY_CONFIG[b_id]
        dist_km = info['distance'] / 1000.0
        idx_pos = np.argmin(np.abs(dists_m - info['distance']))
        target_idx = DEPTH_INDICES[idx_pos]

        col_p = f'pred_tm{target_idx:03d}'
        col_t = f'tm{target_idx:03d}'
        sub_t = df_true[df_true['Buoy_ID'] == b_id][['Time', col_t]].rename(columns={col_t: 'true'})
        sub_n = df_no[df_no['Buoy_ID'] == b_id][['Time', col_p]].rename(columns={col_p: 'no'})
        sub_s = df_sy[df_sy['Buoy_ID'] == b_id][['Time', col_p]].rename(columns={col_p: 'sy'})

        m_full_no = pd.merge(sub_t, sub_n, on='Time').dropna()
        m_full_sy = pd.merge(sub_t, sub_s, on='Time').dropna()
        r2_no = r2_score(m_full_no['true'], m_full_no['no']) if len(m_full_no) > 10 else np.nan
        r2_sy = r2_score(m_full_sy['true'], m_full_sy['sy']) if len(m_full_sy) > 10 else np.nan

        m_plot = sub_t.merge(sub_n, on='Time').merge(sub_s, on='Time').dropna()
        m_plot = m_plot[m_plot['Time'].dt.month.isin([1])].sort_values('Time')

        ax.scatter(m_plot['Time'], m_plot['true'], color=COLOR_OBSERVATION, s=15, alpha=0.7,
                   label='SWAN', zorder=1)
        ax.plot(m_plot['Time'], m_plot['no'], color=COLOR_NO_SYNTHETIC, lw=2.0, alpha=0.9,
                label='SWAN-T', zorder=2)
        ax.plot(m_plot['Time'], m_plot['sy'], color=COLOR_WITH_SYNTHETIC, lw=2.0, alpha=0.9,
                label='SWAN-T (trained with artificial profiles)', zorder=3)

        buoy_display_name = NAME_MAPPING[info['name']]
        ax.set_title(f'{letter} {buoy_display_name} (Tm, {dist_km:.2f} km till shore)',
                     loc='left', fontweight='bold', fontsize=16)

        base_x, base_y = 0.97, 0.96
        line_height = 0.05
        ax.text(base_x, base_y, f"Overall R²(2020):", transform=ax.transAxes,
                fontsize=13, fontweight='bold', ha='right', va='top', color='#2C3E50',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=5))
        ax.text(base_x, base_y - line_height, f"SWAN-T: {r2_no:.2f}", transform=ax.transAxes,
                fontsize=13, fontweight='bold', ha='right', va='top', color=COLOR_NO_SYNTHETIC)
        ax.text(base_x, base_y - 2 * line_height, f"SWAN-T+Aug: {r2_sy:.2f}", transform=ax.transAxes,
                fontsize=13, fontweight='bold', ha='right', va='top', color=COLOR_WITH_SYNTHETIC)

        ax.set_ylabel('Mean Wave Period (Tm, s)', fontweight='bold', fontsize=14)
        ax.set_xlabel('Time (2020)', fontweight='bold', fontsize=14)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        ax.grid(True, alpha=0.3, ls='--')
        if i == 0:
            ax.legend(loc='upper left', fontsize=10, frameon=True)

    plt.savefig('Final_Fig_TM_Temporal_All.png', dpi=300, bbox_inches='tight')
    plt.close()


def calculate_paper_stats(df_no, df_sy, df_true):
    dists_km = np.array(DEPTH_INDICES) * GRID_SPACING / 1000.0

    def get_r2_at_idx(df_p, df_t, idx, buoy_id):
        col_p = f'pred_hs{idx:03d}'
        col_t = f'hs{idx:03d}'
        m = pd.merge(
            df_p[df_p['Buoy_ID'] == buoy_id][['Time', col_p]],
            df_t[df_t['Buoy_ID'] == buoy_id][['Time', col_t]], on='Time'
        ).dropna()
        return r2_score(m[col_t], m[col_p]) if len(m) > 10 else np.nan

    indices_4_7 = np.array(DEPTH_INDICES)[(dists_km >= 2.0) & (dists_km <= 7.0)]
    r2_no_vals = [get_r2_at_idx(df_no, df_true, i, 'Fxs_waves') for i in indices_4_7]
    r2_sy_vals = [get_r2_at_idx(df_sy, df_true, i, 'Fxs_waves') for i in indices_4_7]
    print(f"\n[Fxs 4-7km] Mean improvement: +{np.nanmean(r2_sy_vals) - np.nanmean(r2_no_vals):.4f}")

    deltas = []
    indices_1_4 = np.array(DEPTH_INDICES)[(dists_km >= 1.0) & (dists_km <= 4.0)]
    for b in ['WBy_waves', 'PBy_waves', 'Hrn_waves']:
        deltas.append(np.nanmean(
            [get_r2_at_idx(df_sy, df_true, i, b) - get_r2_at_idx(df_no, df_true, i, b)
             for i in indices_1_4]
        ))
    print(f"[Others 1-4km] Improvement range: {min(deltas):.3f} - {max(deltas):.3f}\n")


if __name__ == "__main__":
    df_sy    = load_transformer_data(EXP_NAME_SYNTHETIC)
    df_no    = load_transformer_data(EXP_NAME_REAL_ONLY)
    df_sw_hs = load_swan_data(FILE_SWAN_HS)
    df_sw_tm = load_swan_data(FILE_SWAN_TM)

    if df_sy is not None and df_no is not None and df_sw_hs is not None:
        calculate_paper_stats(df_no, df_sy, df_sw_hs)
        plot_hs_figure_1(df_no, df_sy, df_sw_hs)
        plot_hs_figure_2(df_no, df_sy, df_sw_hs)

    if df_sy is not None and df_no is not None and df_sw_tm is not None:
        plot_tm_spatial_all(df_no, df_sy, df_sw_tm)
        plot_tm_temporal_all(df_no, df_sy, df_sw_tm)

    print("\nAll figures saved.")
