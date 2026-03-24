# -*- coding: utf-8 -*-
"""
Figure 3 – SWAN-C bias correction evaluation.
Scatter + residual combined panels (Hs and Tm) and box/violin error analysis.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import gaussian_kde
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from pathlib import Path

warnings.filterwarnings('ignore')

# =========================
# Config
# =========================
class Config:
    # Paths relative to this file; update to point to your result directories
    input_dir = Path("../../../results")
    output_dir = Path(".")

    file_train = "results_2020_TRAIN_EVAL.xlsx"
    file_test  = "results_2021_TEST_EVAL.xlsx"

    buoy_data_file = Path("../../../data/fig3/buoy_data_2021.xlsx")

    VAR_CONFIG = {
        'Hs': {'lims': [-1.5, 1.5], 'unit': 'm'},
        'Tm': {'lims': [-5, 7.5],   'unit': 's'},
    }

    COLORS = {
        'train':            '#EF767A',
        'test':             '#3498db',
        'identity_line':    'black',
        'zero_line':        '#333333',
        'box_edge':         '#333333',
        'median':           'black',
        'mean_marker_face': 'white',
        'mean_marker_edge': 'black',
    }

    BLUE_PALETTE = ['#B6D7E8', '#6DADD1', '#317CB7', '#104680']
    SAMPLE_SIZE  = 10000


# =========================
# Plot style
# =========================
plt.rcParams['font.family']         = 'Arial'
plt.rcParams['mathtext.default']    = 'regular'
plt.rcParams['axes.unicode_minus']  = False
plt.rcParams['font.weight']         = 'bold'
plt.rcParams['axes.labelweight']    = 'bold'
plt.rcParams['axes.titleweight']    = 'bold'
plt.rcParams['font.size']           = 15
plt.rcParams['axes.linewidth']      = 1.5
plt.rcParams['xtick.major.width']   = 1.5
plt.rcParams['ytick.major.width']   = 1.5


# =========================
# Utilities
# =========================
def read_data_file(file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"[Error] Not found: {file_path}")
        return None
    if str(file_path).endswith('.xlsx'):
        return pd.read_excel(file_path)
    if str(file_path).endswith('.csv'):
        return pd.read_csv(file_path)
    return None


def get_kde_curve(data, lims, num_points=200):
    if len(data) < 2:
        return np.linspace(lims[0], lims[1], num_points), np.zeros(num_points)
    try:
        kde = gaussian_kde(data)
        x_grid = np.linspace(lims[0], lims[1], num_points)
        return x_grid, kde(x_grid)
    except Exception:
        return np.linspace(lims[0], lims[1], num_points), np.zeros(num_points)


def sample_data(x, y, resid, n_samples):
    if len(x) > n_samples:
        idx = np.random.choice(len(x), n_samples, replace=False)
        return x[idx], y[idx], resid[idx]
    return x, y, resid


def add_label(ax, label):
    ax.text(-0.08, 1.02, label, transform=ax.transAxes,
            fontsize=24, fontweight='extra bold', va='bottom', ha='right')


# =========================
# Combo Plot (Scatter + Residual)
# =========================
def draw_combo_panel(fig, grid_spec_slot, train_df, test_df, target_col, show_label=None):
    var_conf = Config.VAR_CONFIG.get(target_col, {'lims': [-1, 1], 'unit': '-'})
    lims = var_conf['lims']
    unit = var_conf['unit']

    def process(df):
        true_col = f'True_Bias_{target_col}'
        pred_col = f'Pred_Bias_{target_col}'
        x = -df[pred_col].values
        y = -df[true_col].values
        mask = ~np.isnan(x) & ~np.isnan(y)
        return x[mask], y[mask], (y[mask] - x[mask])

    x_tr, y_tr, r_tr = process(train_df)
    x_te, y_te, r_te = process(test_df)

    def stats(x, y):
        return r2_score(y, x), mean_absolute_error(y, x), np.sqrt(mean_squared_error(y, x))

    r2_tr, mae_tr, rmse_tr = stats(x_tr, y_tr)
    r2_te, mae_te, rmse_te = stats(x_te, y_te)

    x_tr_p, y_tr_p, r_tr_p = sample_data(x_tr, y_tr, r_tr, Config.SAMPLE_SIZE)
    x_te_p, y_te_p, r_te_p = sample_data(x_te, y_te, r_te, Config.SAMPLE_SIZE)

    gs_inner = GridSpecFromSubplotSpec(2, 1, height_ratios=[3, 1],
                                       subplot_spec=grid_spec_slot, hspace=0.10)
    gs_top = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_inner[0],
                                     width_ratios=[5, 1], height_ratios=[1, 5],
                                     wspace=0.05, hspace=0.05)

    ax_kx = fig.add_subplot(gs_top[0, 0])
    ax_sc = fig.add_subplot(gs_top[1, 0], sharex=ax_kx)
    ax_ky = fig.add_subplot(gs_top[1, 1], sharey=ax_sc)

    gs_bot = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_inner[1],
                                     width_ratios=[5, 1], wspace=0.05)
    ax_rs = fig.add_subplot(gs_bot[0, 0], sharex=ax_sc)
    ax_rk = fig.add_subplot(gs_bot[0, 1], sharey=ax_rs)

    ax_sc.scatter(x_tr_p, y_tr_p, c=Config.COLORS['train'], s=20, alpha=0.5,
                  marker='o', ec='white', lw=0.2)
    ax_sc.scatter(x_te_p, y_te_p, c=Config.COLORS['test'],  s=20, alpha=0.5,
                  marker='o', ec='white', lw=0.2)
    ax_sc.plot(lims, lims, '--', color=Config.COLORS['identity_line'], lw=2.0)

    ax_sc.set_ylabel(f'SWH residual\nbetween SWAN model and data ({unit})',
                     fontweight='bold', fontsize=16)
    ax_sc.tick_params(axis='both', which='major', labelsize=15, width=2)
    plt.setp(ax_sc.get_xticklabels(), visible=False)
    ax_sc.set_xlim(lims)
    ax_sc.set_ylim(lims)
    ax_sc.grid(True, ls='--', alpha=0.3)

    legs = [
        Line2D([0], [0], marker='o', color='w', mfc=Config.COLORS['train'],
               label='Train (2020)', ms=10),
        Line2D([0], [0], marker='o', color='w', mfc=Config.COLORS['test'],
               label='Test (2021)', ms=10),
        Line2D([0], [0], color='black', lw=2, ls='--', label='1:1 Line'),
    ]
    ax_sc.legend(handles=legs, loc='upper left', ncol=1, frameon=False,
                 prop={'weight': 'bold', 'size': 14})

    txt_tr = f"Train\n$R²={r2_tr:.2f}$\n$MAE={mae_tr:.2f}$"
    txt_te = f"Test\n$R²={r2_te:.2f}$\n$MAE={mae_te:.2f}$"
    ax_sc.text(0.97, 0.25, txt_tr, transform=ax_sc.transAxes, va='bottom', ha='right',
               c=Config.COLORS['train'], fontweight='bold', fontsize=13,
               bbox=dict(fc='white', alpha=1.0, ec='none', pad=2))
    ax_sc.text(0.97, 0.05, txt_te, transform=ax_sc.transAxes, va='bottom', ha='right',
               c=Config.COLORS['test'], fontweight='bold', fontsize=13,
               bbox=dict(fc='white', alpha=1.0, ec='none', pad=2))

    tx, ty = get_kde_curve(x_tr_p, lims)
    dx, dy = get_kde_curve(x_te_p, lims)
    ax_kx.fill_between(tx, 0, ty, color=Config.COLORS['train'], alpha=0.3)
    ax_kx.plot(tx, ty, c=Config.COLORS['train'], lw=2)
    ax_kx.fill_between(dx, 0, dy, color=Config.COLORS['test'], alpha=0.3)
    ax_kx.plot(dx, dy, c=Config.COLORS['test'], lw=2)
    ax_kx.axis('off')

    tyg, txk = get_kde_curve(y_tr_p, lims)
    dyg, dxk = get_kde_curve(y_te_p, lims)
    ax_ky.fill_betweenx(tyg, 0, txk, color=Config.COLORS['train'], alpha=0.3)
    ax_ky.plot(txk, tyg, c=Config.COLORS['train'], lw=2)
    ax_ky.fill_betweenx(dyg, 0, dxk, color=Config.COLORS['test'], alpha=0.3)
    ax_ky.plot(dxk, dyg, c=Config.COLORS['test'], lw=2)
    ax_ky.axis('off')

    ax_rs.scatter(x_tr_p, r_tr_p, c=Config.COLORS['train'], s=20, alpha=0.5,
                  marker='o', ec='white', lw=0.2)
    ax_rs.scatter(x_te_p, r_te_p, c=Config.COLORS['test'],  s=20, alpha=0.5,
                  marker='o', ec='white', lw=0.2)
    ax_rs.axhline(0, c=Config.COLORS['zero_line'], ls='--', lw=2.0)
    ax_rs.set_ylabel(f'Residual\n(Predict-Target, {unit})', fontweight='bold', fontsize=16)
    ax_rs.set_xlabel(f'SWH residual predict by SWAN-C ({unit})', fontweight='bold', fontsize=16)
    ax_rs.tick_params(axis='both', which='major', labelsize=15, width=2)
    ax_rs.set_xlim(lims)
    ax_rs.set_ylim(lims)
    ax_rs.grid(True, ls='--', alpha=0.3)

    rtx, rty = get_kde_curve(r_tr_p, lims)
    rdx, rdy = get_kde_curve(r_te_p, lims)
    ax_rk.fill_betweenx(rtx, 0, rty, color=Config.COLORS['train'], alpha=0.3)
    ax_rk.plot(rty, rtx, c=Config.COLORS['train'], lw=2)
    ax_rk.fill_betweenx(rdx, 0, rdy, color=Config.COLORS['test'], alpha=0.3)
    ax_rk.plot(rdy, rdx, c=Config.COLORS['test'], lw=2)
    ax_rk.axhline(0, c=Config.COLORS['zero_line'], ls='--', lw=2.0)
    ax_rk.set_ylim(lims)
    ax_rk.axis('off')

    if show_label:
        add_label(ax_kx, show_label)


# =========================
# Box + Violin Panel
# =========================
def draw_box_violin_panel(ax, result_df, buoy_path, show_label=None):
    df_res = result_df.copy()
    df_res['time']     = pd.to_datetime(df_res['Time'].astype(str), format='%Y%m%d%H', errors='coerce')
    df_res['buoy_id']  = df_res['Buoy_ID'].astype(str).str.strip()

    df_raw = read_data_file(buoy_path)
    if df_raw is None:
        return

    df_buoy = pd.DataFrame({
        'id':   df_raw['Buoy_ID'].astype(str).str.strip(),
        'time': pd.to_datetime(df_raw['DateTimeGMT'], errors='coerce'),
        'hs':   pd.to_numeric(df_raw['Hs_m'], errors='coerce'),
    }).dropna()
    df_buoy['time'] = df_buoy['time'].dt.floor('h')

    merged   = df_res.merge(df_buoy, left_on=['buoy_id', 'time'], right_on=['id', 'time'], how='inner')
    df_clean = merged[merged['hs'] > 0.1].copy()

    df_clean['error_ratio'] = (df_clean['True_Bias_Hs'] - df_clean['Pred_Bias_Hs']).abs() / df_clean['hs']

    labels = ['0–1', '1–2', '2–3', '>3']
    df_clean['bin'] = pd.cut(df_clean['hs'], bins=[0, 1, 2, 3, np.inf], labels=labels)

    plot_data, valid_labels, counts = [], [], []
    for lbl in labels:
        data = df_clean[df_clean['bin'] == lbl]['error_ratio'].dropna().values
        if len(data) > 0:
            limit = np.percentile(data, 99.0)
            filtered = data[data <= limit]
            plot_data.append(filtered)
            valid_labels.append(lbl)
            counts.append(len(data))

    if not plot_data:
        return

    indices    = np.arange(len(plot_data))
    box_width  = 0.35
    violin_width = 0.6
    offset     = 0.12
    pos_box    = indices - offset
    pos_violin = indices + offset

    bp = ax.boxplot(plot_data, positions=pos_box, widths=box_width,
                    patch_artist=True, showfliers=False, showmeans=True,
                    boxprops=dict(edgecolor=Config.COLORS['box_edge'], lw=2.0),
                    medianprops=dict(color=Config.COLORS['median'], lw=2.5),
                    meanprops=dict(marker='o', mfc=Config.COLORS['mean_marker_face'],
                                   mec='black', ms=8, mew=1.5))

    colors = Config.BLUE_PALETTE[:len(plot_data)]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)

    vp = ax.violinplot(plot_data, positions=pos_violin, widths=violin_width,
                       showmeans=False, showmedians=False, showextrema=False,
                       points=10000, bw_method='scott')
    for i, body in enumerate(vp['bodies']):
        c = colors[i]
        body.set_facecolor(c)
        body.set_edgecolor(c)
        body.set_alpha(0.6)
        path = body.get_paths()[0]
        verts = path.vertices.copy()
        center = pos_violin[i]
        verts[:, 0] = np.where(verts[:, 0] < center, center, verts[:, 0])
        body.set_verts([verts])

    ax.set_xticks(indices)
    ax.set_xticklabels([f'{l}\n(N={c})' for l, c in zip(valid_labels, counts)],
                       fontsize=15, fontweight='bold')
    ax.set_xlabel('Observed SWH Range (m)', fontweight='bold', fontsize=16)
    ax.set_ylabel('Normalized Absolute Prediction Error (-)', fontweight='bold', fontsize=16)
    ax.tick_params(axis='y', which='major', labelsize=15, width=2)
    ax.tick_params(axis='x', which='major', width=2)
    ax.set_ylim(-0.02, 0.7)
    ax.set_xlim(-0.6, len(indices) - 0.4)
    ax.grid(True, axis='y', ls='--', alpha=0.3)

    legs = [Line2D([0], [0], marker='o', color='w', mfc='white', mec='black',
                   ms=10, mew=1.5, label='Mean value')]
    ax.legend(handles=legs, loc='upper right', frameon=False, prop={'weight': 'bold', 'size': 14})

    if show_label:
        add_label(ax, show_label)


# =========================
# Main
# =========================
def main():
    path_tr = Config.input_dir / Config.file_train
    path_te = Config.input_dir / Config.file_test

    df_tr = read_data_file(path_tr)
    df_te = read_data_file(path_te)

    if df_tr is None or df_te is None:
        print("Data loading failed. Please check paths in Config class.")
        return

    print("Generating Figure 1: Hs Combined...")
    fig1 = plt.figure(figsize=(18, 9))
    gs = GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.18)
    draw_combo_panel(fig1, gs[0], df_tr, df_te, target_col='Hs', show_label='a')
    ax_b = fig1.add_subplot(gs[1])
    draw_box_violin_panel(ax_b, df_te, Config.buoy_data_file, show_label='b')
    out1 = Config.output_dir / "Fig3_Hs_Combined.png"
    fig1.savefig(out1, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    print(f"Saved: {out1}")

    print("Generating Figure 2: Tm Combo...")
    fig2 = plt.figure(figsize=(9, 9))
    gs2 = GridSpec(1, 1)
    draw_combo_panel(fig2, gs2[0], df_tr, df_te, target_col='Tm')
    out2 = Config.output_dir / "Fig3_Tm_Combo.png"
    fig2.savefig(out2, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print(f"Saved: {out2}")

    print("Done.")


if __name__ == "__main__":
    main()
