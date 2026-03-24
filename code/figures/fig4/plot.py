# -*- coding: utf-8 -*-
"""
Figure 4 – SWAN-TC validation against buoy observations (2021–2023).
Scatter plots, bias-ratio boxplots, and annual R² bar charts comparing
SWAN-T, SWAN-TC, and WW3 baseline.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats as scipy_stats
from matplotlib.patches import Patch
from pathlib import Path


# =========================
# Config
# =========================
class TransformerPredictConfig:
    spatial_points  = 64
    depth_indices   = list(range(10, 200, 3))
    grid_spacing    = 50

    # Update to point to your SWAN-TC prediction results directory
    result_dir = Path("../../../results/swan_tc")

    test_months     = [1, 7]
    predict_years   = [2021, 2022, 2023]


class Config:
    wave_height_threshold = 0.1
    spatial_points  = TransformerPredictConfig.spatial_points
    depth_indices   = TransformerPredictConfig.depth_indices
    grid_spacing    = TransformerPredictConfig.grid_spacing
    result_dir      = Path(".")
    predict_years   = TransformerPredictConfig.predict_years
    test_months     = TransformerPredictConfig.test_months


COLORS = {
    'scatter_points':    '#4DBBD5',
    'model_transformer': '#E64B35',
    'model_corrected':   '#4DBBD5',
    'model_ww3':         '#555555',
    'boxplot_edge':      'black',
    'zero_line':         'gray',
    'median_line':       '#000000',
    'grid_line':         'gray',
    'perfect_line':      'black',
}

# Buoy distances to shore (m)
buoy_distances = {
    'Hrn_waves': 7015.19, 'Fxs_waves': 3308.10,
    'PBy_waves': 5294.04, 'WBy_waves': 1198.36,
}

SELECTED_BUOYS = ['Hrn_waves', 'Fxs_waves', 'PBy_waves', 'WBy_waves']

plt.rcParams.update({
    'font.family':      'Arial',
    'font.size':        14,
    'axes.labelsize':   14,
    'axes.titlesize':   16,
    'xtick.labelsize':  14,
    'ytick.labelsize':  14,
    'legend.fontsize':  14,
    'axes.linewidth':   0.8,
    'axes.edgecolor':   'black',
    'axes.axisbelow':   True,
    'axes.grid':        True,
    'grid.linewidth':   0.4,
    'grid.alpha':       0.3,
    'grid.color':       COLORS['grid_line'],
    'grid.linestyle':   '--',
    'xtick.direction':  'in',
    'ytick.direction':  'in',
    'xtick.top':        True,
    'ytick.right':      True,
    'figure.dpi':       300,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'axes.facecolor':   'white',
    'figure.facecolor': 'white',
})

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


# =========================
# Statistics
# =========================
def calculate_statistics(obs, pred):
    if len(obs) <= 1:
        return {'r2': np.nan, 'mae': np.nan, 'n': 0}
    obs, pred = np.array(obs), np.array(pred)
    valid = ~(np.isnan(obs) | np.isnan(pred))
    obs, pred = obs[valid], pred[valid]
    if len(obs) <= 1:
        return {'r2': np.nan, 'mae': np.nan, 'n': 0}
    return {'r2': r2_score(obs, pred), 'mae': mean_absolute_error(obs, pred), 'n': len(obs)}


def calculate_point_density(x, y):
    points = np.vstack([x, y]).T
    try:
        kde = scipy_stats.gaussian_kde(points.T)
        density = kde(points.T)
        return 0.3 + (0.8 - 0.3) * (density - density.min()) / (density.max() - density.min() + 1e-10)
    except Exception:
        return np.full(len(x), 0.5)


def add_panel_label(ax, label):
    ax.text(-0.05, 1.05, label, transform=ax.transAxes,
            fontsize=20, fontweight='bold', va='bottom', ha='right')


# =========================
# Data loading
# =========================
def load_data_for_year(year):
    # Update these paths to your data directories
    buoy_path = Path(f"../../../data/fig4_fig5/buoy_data_{year}.xlsx")

    pred_base = TransformerPredictConfig.result_dir / f"Final_Pred_{year}"

    ww3_dir = Path("../../../data/fig4_fig5/ww3")
    ww3_files = {
        'hs_jan': ww3_dir / f"wave_height_uk_ww3_jan_{year}.xlsx",
        'hs_jul': ww3_dir / f"wave_height_uk_ww3_jul_{year}.xlsx",
        'tm_jan': ww3_dir / f"wave_period_uk_ww3_jan_{year}.xlsx",
        'tm_jul': ww3_dir / f"wave_period_uk_ww3_jul_{year}.xlsx",
    }

    data = {k: pd.DataFrame()
            for k in ['buoy', 'uncorrected', 'corrected_hs', 'corrected_tm', 'ww3_hs', 'ww3_tm']}

    cols_id   = ['Buoy_ID', 'buoy_id', 'id', 'station_id']
    cols_time = ['DateTimeGMT', 'time', 'Time', 'date']

    def fast_filter(df, id_cands, time_cands, var_type=None, suffix="", time_fmt=None):
        if df.empty:
            return pd.DataFrame()
        id_col = next((c for c in id_cands if c in df.columns), None)
        if not id_col:
            return pd.DataFrame()
        df.rename(columns={id_col: 'id'}, inplace=True)
        df = df[df['id'].isin(SELECTED_BUOYS)].copy()
        if df.empty:
            return pd.DataFrame()
        time_col = next((c for c in time_cands if c in df.columns), None)
        if time_col:
            df.rename(columns={time_col: 'time'}, inplace=True)
            df['Time_dt'] = (pd.to_datetime(df['time'], format=time_fmt, errors='coerce')
                             if time_fmt else pd.to_datetime(df['time'], errors='coerce'))
            df['Time_dt'] = df['Time_dt'].dt.floor('h')
            df = df[df['Time_dt'].dt.month.isin(Config.test_months)].reset_index(drop=True)
        else:
            return pd.DataFrame()
        if var_type:
            new_cols = {c: c.replace(f"{var_type}_", f"pred_{var_type}") + suffix
                        for c in df.columns if c.startswith(f"{var_type}_")}
            df.rename(columns=new_cols, inplace=True)
        return df

    if buoy_path.exists():
        try:
            df = pd.read_excel(buoy_path, engine='openpyxl')
            data['buoy'] = fast_filter(df, cols_id, cols_time)
            if 'Tm02_s' in data['buoy'].columns:
                data['buoy']['Tm02_s'] = pd.to_numeric(data['buoy']['Tm02_s'], errors='coerce')
        except Exception as e:
            logger.error(f"Failed to load buoy data: {e}")

    file_map = [
        (f"{pred_base}_Hs_Raw.csv",       'uncorrected',   'hs', ""),
        (f"{pred_base}_Hs_Corrected.csv", 'corrected_hs',  'hs', "_corrected"),
        (f"{pred_base}_Tm_Corrected.csv", 'corrected_tm',  'tm', "_corrected"),
    ]
    for f, k, v, s in file_map:
        if Path(f).exists():
            try:
                data[k] = fast_filter(pd.read_csv(f), cols_id, cols_time, var_type=v, suffix=s)
            except Exception as e:
                logger.error(f"Failed to load prediction {f}: {e}")

    def load_ww3(paths, key):
        dfs = []
        for p in paths:
            if Path(str(p)).exists():
                try:
                    df = pd.read_excel(p, engine='openpyxl')
                    if 'time' in df.columns:
                        df['time'] = df['time'].astype(str).str.replace(r'\.0$', '', regex=True)
                    df_proc = fast_filter(df, cols_id, cols_time, time_fmt='%Y%m%d%H')
                    for c in ['hs', 'swh', 'tm02', 'm02']:
                        target = next((col for col in df_proc.columns if col.lower() == c), None)
                        if target:
                            df_proc[target] = pd.to_numeric(df_proc[target], errors='coerce')
                    dfs.append(df_proc)
                except Exception as e:
                    logger.error(f"Error loading WW3 {p}: {e}")
        if dfs:
            data[key] = pd.concat(dfs, ignore_index=True)

    load_ww3([ww3_files['hs_jan'], ww3_files['hs_jul']], 'ww3_hs')
    load_ww3([ww3_files['tm_jan'], ww3_files['tm_jul']], 'ww3_tm')
    return data


# =========================
# Plotting
# =========================
def plot_comparison(data):
    os.makedirs(Config.result_dir, exist_ok=True)

    merged_df = data['buoy'].copy()

    def clean_merge(left_df, right_df, right_name=""):
        if right_df.empty:
            return left_df
        exclude_cols = ['time', 'Time', 'Buoy_ID', 'buoy_id', 'Unnamed: 0']
        cols_to_use = [c for c in right_df.columns
                       if c in ['Time_dt', 'id'] or c not in exclude_cols]
        suffix = f"_{right_name}" if right_name else ""
        try:
            return pd.merge(left_df, right_df[cols_to_use], on=['Time_dt', 'id'],
                            how='inner', suffixes=('', suffix))
        except pd.errors.MergeError:
            right_subset = right_df[cols_to_use].copy()
            for col in right_subset.columns:
                if col not in ['Time_dt', 'id'] and col in left_df.columns:
                    right_subset.rename(columns={col: f"{col}{suffix}"}, inplace=True)
            return pd.merge(left_df, right_subset, on=['Time_dt', 'id'], how='inner')

    if not data['ww3_hs'].empty:
        hs_col = next((c for c in data['ww3_hs'].columns
                       if c.lower() in ['hs', 'swh', 'hs_m']), None)
        cols = ['id', 'Time_dt'] + ([hs_col] if hs_col else [])
        merged_df = clean_merge(merged_df, data['ww3_hs'][cols], "ww3hs")

    if not data['ww3_tm'].empty:
        tm_col = next((c for c in data['ww3_tm'].columns
                       if c.lower() in ['tm02', 'm02', 'tm']), None)
        cols = ['id', 'Time_dt'] + ([tm_col] if tm_col else [])
        merged_df = clean_merge(merged_df, data['ww3_tm'][cols], "ww3tm")

    if not data['uncorrected'].empty:
        merged_df = clean_merge(merged_df, data['uncorrected'], "unc")
    if not data['corrected_hs'].empty:
        merged_df = clean_merge(merged_df, data['corrected_hs'], "cor_hs")
    if not data['corrected_tm'].empty:
        merged_df = clean_merge(merged_df, data['corrected_tm'], "cor_tm")

    if merged_df.empty:
        logger.error("Merged dataframe is empty!")
        return

    depth_indices = Config.depth_indices
    x_distances   = Config.grid_spacing * np.array(depth_indices)
    cols_hs_unc   = [f'pred_hs{idx:03d}'            for idx in depth_indices]
    cols_hs_cor   = [f'pred_hs{idx:03d}_corrected'  for idx in depth_indices]
    cols_tm_cor   = [f'pred_tm{idx:03d}_corrected'  for idx in depth_indices]

    for c in ['val_hs_obs', 'val_hs_ww3']:
        merged_df[c] = np.nan
    for c in ['val_hs_unc', 'val_hs_cor', 'val_tm_cor', 'val_tm_obs', 'val_tm_ww3']:
        merged_df[c] = np.nan

    for col in ['Hs_m', 'Hs_m_buoy']:
        if col in merged_df.columns:
            merged_df['val_hs_obs'] = merged_df[col]
            break

    ww3_hs_cands = [c for c in merged_df.columns if 'ww3hs' in c]
    if ww3_hs_cands:
        merged_df['val_hs_ww3'] = merged_df[ww3_hs_cands[0]]

    if 'Tm02_s' in merged_df.columns:
        merged_df['val_tm_obs'] = merged_df['Tm02_s']
    ww3_tm_cands = [c for c in merged_df.columns if 'ww3tm' in c]
    if ww3_tm_cands:
        merged_df['val_tm_ww3'] = merged_df[ww3_tm_cands[0]]

    valid_indices = []
    for idx, row in merged_df.iterrows():
        bid  = row['id']
        dist = buoy_distances.get(bid, 0)
        i    = np.argmin(np.abs(x_distances - dist))
        if i >= len(cols_hs_unc):
            continue
        merged_df.at[idx, 'val_hs_unc'] = row.get(cols_hs_unc[i], np.nan)
        merged_df.at[idx, 'val_hs_cor'] = row.get(cols_hs_cor[i], np.nan)
        merged_df.at[idx, 'val_tm_cor'] = row.get(cols_tm_cor[i], np.nan)
        valid_indices.append(idx)

    df_clean = merged_df.loc[valid_indices].copy()
    for c in ['val_hs_obs', 'val_hs_ww3', 'val_hs_unc', 'val_hs_cor']:
        df_clean[c] = pd.to_numeric(df_clean[c], errors='coerce')
    df_clean.dropna(subset=['val_hs_obs', 'val_hs_ww3', 'val_hs_unc', 'val_hs_cor'], inplace=True)
    df_clean = df_clean[df_clean['val_hs_obs'] > Config.wave_height_threshold]

    if len(df_clean) == 0:
        logger.error("No valid data after filtering!")
        return

    all_hs = np.concatenate([df_clean['val_hs_unc'].values, df_clean['val_hs_cor'].values,
                              df_clean['val_hs_ww3'].values, df_clean['val_hs_obs'].values])
    vmin_hs, vmax_hs = np.nanmin(all_hs), np.nanmax(all_hs)
    pad   = 0.05 * (vmax_hs - vmin_hs)
    limit = (max(0, vmin_hs - pad), vmax_hs + pad)

    fig = plt.figure(figsize=(14, 10))
    gs_outer  = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.25,
                                 left=0.06, right=0.98, top=0.92, bottom=0.08)
    gs_top    = gs_outer[0].subgridspec(1, 4, wspace=0.20)
    gs_bottom = gs_outer[1].subgridspec(1, 2, wspace=0.20, width_ratios=[1, 1])

    ax_s1 = fig.add_subplot(gs_top[0])
    ax_s2 = fig.add_subplot(gs_top[1])
    ax_s3 = fig.add_subplot(gs_top[2])
    ax_s4 = fig.add_subplot(gs_top[3])
    ax_box = fig.add_subplot(gs_bottom[0])
    ax_bar = fig.add_subplot(gs_bottom[1])

    text_props = dict(facecolor='none', edgecolor='none', alpha=1.0)

    def plot_scatter(ax, x, y, label, lx, ly, fixed_lims=None, show_y=True, show_yticks=True):
        mask = ~(np.isnan(x) | np.isnan(y))
        xc, yc = x[mask], y[mask]
        if len(xc) < 2:
            return
        s = calculate_statistics(yc, xc)
        vmin, vmax = (fixed_lims if fixed_lims else
                      (max(0, min(xc.min(), yc.min()) - 0.05 * (max(xc.max(), yc.max()) - min(xc.min(), yc.min()))),
                       max(xc.max(), yc.max()) + 0.05 * (max(xc.max(), yc.max()) - min(xc.min(), yc.min()))))
        density = calculate_point_density(xc, yc)
        ax.scatter(xc, yc, c=COLORS['scatter_points'], s=15, alpha=density,
                   edgecolors='black', linewidths=0.5)
        ax.plot([vmin, vmax], [vmin, vmax], COLORS['perfect_line'], linestyle='--', lw=2.0)
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_aspect('equal')
        ax.set_xlabel(lx, fontweight='bold')
        if show_y:
            ax.set_ylabel(ly, fontweight='bold')
        else:
            ax.set_ylabel("")
        if not show_yticks:
            ax.tick_params(labelleft=False)
        add_panel_label(ax, label)
        ax.text(0.05, 0.95, f"R² = {s['r2']:.2f}\nMAE = {s['mae']:.2f}",
                transform=ax.transAxes, va='top', bbox=text_props, fontsize=14, fontweight='bold')

    plot_scatter(ax_s1, df_clean['val_hs_unc'].values, df_clean['val_hs_obs'].values,
                 'a', 'SWAN-T predicted SWH (m)', 'Observed SWH (m)', limit)
    plot_scatter(ax_s2, df_clean['val_hs_cor'].values, df_clean['val_hs_obs'].values,
                 'b', 'SWAN-TC predicted SWH (m)', 'Observed SWH (m)', limit, False, False)
    plot_scatter(ax_s3, df_clean['val_hs_ww3'].values, df_clean['val_hs_obs'].values,
                 'c', 'WW3 SWH (m)', 'Observed SWH (m)', limit, False, False)

    tm_sub = df_clean[['val_tm_cor', 'val_tm_obs']].dropna()
    plot_scatter(ax_s4, tm_sub['val_tm_cor'].values, tm_sub['val_tm_obs'].values,
                 'd', 'SWAN-TC predicted Tm02 (s)', 'Observed Tm02 (s)')

    obs_arr = df_clean['val_hs_obs'].values
    r_u = (df_clean['val_hs_unc'].values - obs_arr) / (obs_arr + 1e-6)
    r_c = (df_clean['val_hs_cor'].values - obs_arr) / (obs_arr + 1e-6)
    r_v = (df_clean['val_hs_ww3'].values - obs_arr) / (obs_arr + 1e-6)

    bins      = [0, 1, 2, 3, np.inf]
    labels    = ['<1 m', '1–2 m', '2–3 m', '>3 m']
    bin_idxs  = np.digitize(obs_arr, bins, right=True)
    pos       = np.arange(len(labels))
    width     = 0.2
    box_cols  = [COLORS['model_transformer'], COLORS['model_corrected'], COLORS['model_ww3']]

    for i in range(len(labels)):
        mask = bin_idxs == (i + 1)
        if not np.any(mask):
            continue
        group = [r_u[mask], r_c[mask], r_v[mask]]
        curr  = [pos[i] - width, pos[i], pos[i] + width]
        for j, (dat, p) in enumerate(zip(group, curr)):
            if len(dat) > 0:
                bp = ax_box.boxplot([dat], positions=[p], widths=width * 0.9,
                                    patch_artist=True, showfliers=False)
                for patch in bp['boxes']:
                    patch.set_facecolor(box_cols[j])
                    patch.set_alpha(0.8)
                    patch.set_edgecolor(COLORS['boxplot_edge'])
                for median in bp['medians']:
                    median.set_color('black')

    ax_box.axhline(0, color='gray', linestyle='--', lw=1.5)
    ax_box.set_xticks(pos)
    ax_box.set_xticklabels(labels, fontweight='bold')
    ax_box.set_xlabel('Observed Significant Wave Height Range (m)', fontweight='bold')
    ax_box.set_ylabel('Bias Ratio', fontweight='bold')
    add_panel_label(ax_box, 'e')
    ax_box.set_ylim(-1, 1)
    ax_box.legend(handles=[Patch(facecolor=c, label=l)
                            for c, l in zip(box_cols, ['SWAN-T', 'SWAN-TC', 'WW3 Model'])],
                  loc='upper right', fontsize=14, framealpha=0.95, edgecolor='black')

    years = Config.predict_years
    yearly_r2 = {'SWAN-T': [], 'SWAN-TC': [], 'WW3': []}
    for y in years:
        sub = df_clean[df_clean['Time_dt'].dt.year == y]
        if len(sub) > 10:
            yearly_r2['SWAN-T'].append(r2_score(sub['val_hs_obs'], sub['val_hs_unc']))
            yearly_r2['SWAN-TC'].append(r2_score(sub['val_hs_obs'], sub['val_hs_cor']))
            yearly_r2['WW3'].append(r2_score(sub['val_hs_obs'], sub['val_hs_ww3']))
        else:
            for m in yearly_r2:
                yearly_r2[m].append(0)

    x = np.arange(len(years))
    pw = 0.22
    bw = 0.20
    ax_bar.bar(x - pw, yearly_r2['SWAN-T'],  width=bw, label='SWAN-T',
               color=COLORS['model_transformer'], edgecolor='black')
    ax_bar.bar(x,      yearly_r2['SWAN-TC'], width=bw, label='SWAN-TC',
               color=COLORS['model_corrected'],   edgecolor='black')
    ax_bar.bar(x + pw, yearly_r2['WW3'],     width=bw, label='WW3 Model',
               color=COLORS['model_ww3'],         edgecolor='black')

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(years, fontweight='bold')
    ax_bar.set_xlabel('Year', fontweight='bold')
    ax_bar.set_ylabel('R²', fontweight='bold')
    add_panel_label(ax_bar, 'f')
    ax_bar.set_ylim(0.6, 1.0)
    ax_bar.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax_bar.legend(loc='upper center', ncol=3, fontsize=14, framealpha=0.95, edgecolor='black')

    save_path = Config.result_dir / "Figure4_SWAN_TC_Validation.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    logger.info(f"Figure saved: {save_path}")


def main():
    data_list = [load_data_for_year(y) for y in Config.predict_years]
    combined  = {k: pd.concat([d[k] for d in data_list], ignore_index=True)
                 for k in data_list[0].keys()}

    if not combined['buoy'].empty:
        plot_comparison(combined)
    else:
        logger.error("No data loaded!")


if __name__ == "__main__":
    main()
