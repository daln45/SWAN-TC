# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
import re
import os
import matplotlib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from catboost import CatBoostRegressor
import random
import pickle
import logging
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('./results_multi_bias_pred', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./results_multi_bias_pred/predict_swan_c.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

random.seed(42)
np.random.seed(42)


class Config:
    """预测配置类"""

    _BASE_DIR = Path(__file__).resolve().parent.parent.parent   # repo root
    _DATA_DIR = _BASE_DIR / "data" / "uk"
    _WEIGHTS_DIR = _BASE_DIR / "weights"

    data_files_2021 = {
        "depth": str(_DATA_DIR / "swan_depth_2021.csv"),
        "wind": str(_DATA_DIR / "wind_wave_initial_data_2021.csv"),
        "bias": None
    }
    data_files_2022 = {
        "depth": str(_DATA_DIR / "swan_depth_2022.csv"),
        "wind": str(_DATA_DIR / "wind_wave_initial_data_2022.csv"),
        "bias": None
    }
    data_files_2023 = {
        "depth": str(_DATA_DIR / "swan_depth_2023.csv"),
        "wind": str(_DATA_DIR / "wind_wave_initial_data_2023.csv"),
        "bias": None
    }

    result_dir = "./results_multi_bias_pred"
    model_path = str(_WEIGHTS_DIR / "best_multi_bias_model.cbm")
    scaler_path = str(_WEIGHTS_DIR / "scaler_multi_bias.pkl")

    all_buoys = [
        "Bos_waves", "HgI_waves", "ChP_waves", "Mlf_waves", "BkB_waves", "Clv_waves", "Csl_waves", "Dwl_waves",
        "Flk_waves", "GdS_waves", "Hpg_waves", "LoB_waves", "Nbg_waves", "Plv_waves", "Pnz_waves", "Prp_waves",
        "RhF_waves", "Rst_waves", "SBy_waves", "Sca_waves", "StB_waves", "Wtb_waves", "Bdf_waves", "Hrn_waves",
        "Fxs_waves", "PBy_waves", "WBy_waves"
    ]

    coord_cols = ['depth']
    depth_indices = [200]
    grid_spacing = 50

    test_year = 2021
    test_months = list(range(1, 13))
    predict_years = [2022, 2023]


class SpatialCoordManager:
    @classmethod
    def load_coordinates(cls, depth_df: pd.DataFrame) -> dict:
        if not pd.api.types.is_datetime64_any_dtype(depth_df['time']):
            depth_df['time'] = pd.to_datetime(depth_df['time'], format='%Y%m%d%H', errors='coerce')

        depth_df = depth_df.dropna(subset=['time'])
        depth_df['time'] = depth_df['time'].dt.floor('h')

        depth_map = {}
        for buoy_id, group in depth_df.groupby('id'):
            depth_map[buoy_id] = {}
            temp_dict = group.set_index('time')['depth200'].to_dict()
            for t, d in temp_dict.items():
                depth_map[buoy_id][t] = np.array([d], dtype=np.float32)

        x = Config.grid_spacing * np.array([200], dtype=np.float32)
        y = np.zeros(1, dtype=np.float32)
        return {'x': x, 'y': y, 'depth_map': depth_map}


class SimpleFeatureProcessor:
    @staticmethod
    def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col in ['tide']:
            if col not in df.columns:
                df[col] = 0.0

        feature_cols = ['swh', 'wind_speed', 'wind_direction', 'mwd', 'mwp', 'tide']
        for col in feature_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)

        if 'wind_direction' in df.columns:
            wd_rad = np.deg2rad(df['wind_direction'])
            df['sin_wd'] = np.sin(wd_rad).astype(np.float32)
            df['cos_wd'] = np.cos(wd_rad).astype(np.float32)

        if 'mwd' in df.columns:
            mwd_rad = np.deg2rad(df['mwd'])
            df['sin_mwd'] = np.sin(mwd_rad).astype(np.float32)
            df['cos_mwd'] = np.cos(mwd_rad).astype(np.float32)

        return df


class HierarchicalScaler:
    def __init__(self):
        self.bias_hs_scaler = StandardScaler()
        self.bias_tm_scaler = StandardScaler()
        self.depth_scaler = StandardScaler()
        self.feature_scalers = {}

    def fit(self, df: pd.DataFrame):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'bias_hs' in df.columns:
            df['bias_hs'] = self.bias_hs_scaler.transform(df[['bias_hs']].fillna(0))
        if 'bias_tm' in df.columns:
            df['bias_tm'] = self.bias_tm_scaler.transform(df[['bias_tm']].fillna(0))
        if 'depth200' in df.columns:
            df['depth200'] = self.depth_scaler.transform(df[['depth200']].fillna(0))
        for feat, scaler in self.feature_scalers.items():
            if feat in df.columns:
                df[feat] = scaler.transform(df[[feat]].fillna(0))
        return df

    def inverse_transform_targets(self, preds_hs, preds_tm):
        hs_orig = self.bias_hs_scaler.inverse_transform(preds_hs.reshape(-1, 1)).squeeze()
        tm_orig = self.bias_tm_scaler.inverse_transform(preds_tm.reshape(-1, 1)).squeeze()
        return hs_orig, tm_orig


class MultiBiasPredictDataset(Dataset):
    def __init__(self, year, months, buoy_ids=None, require_bias=True):
        self.year = year
        self.months = months
        self.buoy_ids = buoy_ids if buoy_ids is not None else Config.all_buoys
        self.require_bias = require_bias

        try:
            self._load_spatial_system()
            self.df = self._load_and_merge_data()
            self.df = SimpleFeatureProcessor.add_basic_features(self.df)

            req_cols = ['swh', 'wind_speed', 'mwp', 'tide', 'sin_wd', 'cos_wd', 'sin_mwd', 'cos_mwd']
            missing = [c for c in req_cols if c not in self.df.columns]
            if missing:
                raise ValueError(f"缺失必要特征列: {missing}")

            logger.info(f"Dataset Loaded: Year {year}, Samples: {len(self.df)}")
        except Exception as e:
            logger.error(f"Dataset load error: {e}")
            raise

    def __len__(self):
        return len(self.df)

    def _get_files_config(self):
        if self.year == 2021: return Config.data_files_2021
        if self.year == 2022: return Config.data_files_2022
        if self.year == 2023: return Config.data_files_2023
        return {}

    def _load_spatial_system(self):
        files = self._get_files_config()
        depth_df = pd.read_csv(files["depth"])
        depth_df = depth_df[depth_df['id'].isin(self.buoy_ids)]
        self.coord_system = SpatialCoordManager.load_coordinates(depth_df)

    def _load_and_merge_data(self) -> pd.DataFrame:
        files = self._get_files_config()

        depth = pd.read_csv(files['depth'], dtype={'time': str})
        wind = pd.read_csv(files['wind'], dtype={'time': str})

        for df in [depth, wind]:
            df['time'] = pd.to_datetime(df['time'], format='%Y%m%d%H', errors='coerce')
            df['time'] = df['time'].dt.floor('h')

        depth = depth[depth['time'].dt.month.isin(self.months)]
        wind = wind[wind['time'].dt.month.isin(self.months)]

        merged = depth.merge(wind, on=['id', 'time'], how='inner')

        if files.get('bias') and self.require_bias:
            bias = pd.read_csv(files['bias'], dtype={'time': str})
            t_formatted = pd.to_datetime(bias['time'], format='%Y%m%d%H', errors='coerce')
            if t_formatted.isna().any():
                mask = t_formatted.isna()
                t_formatted.loc[mask] = pd.to_datetime(bias.loc[mask, 'time'], errors='coerce')
            bias['time'] = t_formatted.dt.floor('h')

            if 'bias_hs' not in bias.columns and 'bias' in bias.columns:
                bias['bias_hs'] = bias['bias']
                bias['bias_tm'] = 0.0

            merged = merged.merge(bias, on=['id', 'time'], how='inner')
        else:
            merged['bias_hs'] = 0.0
            merged['bias_tm'] = 0.0

        merged = merged[merged['id'].isin(self.buoy_ids)]
        return merged

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        try:
            depth = self.coord_system['depth_map'][row['id']][row['time']]
        except KeyError:
            depth = np.array([row['depth200']], dtype=np.float32)

        # 物理特征: [swh, wind_speed, mwp, tide, sin_wd, cos_wd, sin_mwd, cos_mwd]
        phy_feats = np.array([
            row['swh'], row['wind_speed'], row['mwp'], row['tide'],
            row['sin_wd'], row['cos_wd'], row['sin_mwd'], row['cos_mwd']
        ], dtype=np.float32)

        targets = np.array([row['bias_hs'], row['bias_tm']], dtype=np.float32)

        return phy_feats, depth, targets, row['id'], row['time'].strftime('%Y%m%d%H')


def prepare_data_for_prediction(dataset):
    X_list, y_list = [], []
    ids, times = [], []

    for i in range(len(dataset)):
        p_feat, d_feat, target, bid, t = dataset[i]
        combined = np.concatenate([p_feat, d_feat])
        X_list.append(combined)
        y_list.append(target)
        ids.append(bid)
        times.append(t)

    X = np.array(X_list)
    y = np.array(y_list)

    feat_names = [
        'swh', 'ws', 'mwp', 'tide',
        'sin_wd', 'cos_wd', 'sin_mwd', 'cos_mwd',
        'depth200'
    ]
    X_df = pd.DataFrame(X, columns=feat_names)

    return X_df, y, ids, times


def evaluate_test_set():
    logger.info("=" * 30 + " 开始测试集评估 (2021) " + "=" * 30)

    if not os.path.exists(Config.model_path):
        logger.error("模型文件未找到！")
        return

    model = CatBoostRegressor()
    model.load_model(Config.model_path)

    with open(Config.scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    ds = MultiBiasPredictDataset(Config.test_year, Config.test_months, require_bias=True)
    if ds.df.empty:
        return

    ds.df = scaler.transform(ds.df)
    X, y_scaled, ids, times = prepare_data_for_prediction(ds)

    preds_scaled = model.predict(X)

    pred_hs, pred_tm = scaler.inverse_transform_targets(preds_scaled[:, 0], preds_scaled[:, 1])
    true_hs, true_tm = scaler.inverse_transform_targets(y_scaled[:, 0], y_scaled[:, 1])

    if not np.all(true_hs == 0):
        calculate_metrics(true_hs, pred_hs, "Hs")
        calculate_metrics(true_tm, pred_tm, "Tm")
        plot_r2_bar(ids, true_hs, pred_hs, "Hs Bias", Config.test_year)
        plot_r2_bar(ids, true_tm, pred_tm, "Tm Bias", Config.test_year)
    else:
        logger.warning("测试集缺少真实 Bias 标签，跳过指标计算。")

    save_results_excel(ids, times, true_hs, pred_hs, true_tm, pred_tm,
                       year=Config.test_year, suffix="TEST_EVAL")


def predict_future_years():
    logger.info("=" * 30 + " 开始未来年份预测 " + "=" * 30)

    if not os.path.exists(Config.model_path):
        return

    model = CatBoostRegressor()
    model.load_model(Config.model_path)

    with open(Config.scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    for year in Config.predict_years:
        logger.info(f"正在预测 {year} 年...")
        try:
            ds = MultiBiasPredictDataset(year, Config.test_months, require_bias=False)
            if ds.df.empty:
                continue

            ds.df = scaler.transform(ds.df)
            X, _, ids, times = prepare_data_for_prediction(ds)

            preds_scaled = model.predict(X)

            pred_hs, pred_tm = scaler.inverse_transform_targets(preds_scaled[:, 0], preds_scaled[:, 1])

            zeros = np.zeros_like(pred_hs)
            save_results_excel(ids, times, zeros, pred_hs, zeros, pred_tm,
                               year=year, suffix="PREDICTION")

        except Exception as e:
            logger.error(f"{year}年预测失败: {e}")


def calculate_metrics(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    logger.info(f"[{name}] RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return {'rmse': rmse, 'r2': r2}


def save_results_excel(ids, times, true_hs, pred_hs, true_tm, pred_tm, year, suffix):
    df = pd.DataFrame({
        'Buoy_ID': ids,
        'Time': times,
        'True_Bias_Hs': true_hs,
        'Pred_Bias_Hs': pred_hs,
        'True_Bias_Tm': true_tm,
        'Pred_Bias_Tm': pred_tm
    })

    if suffix == "PREDICTION":
        df = df.drop(columns=['True_Bias_Hs', 'True_Bias_Tm'])

    path = f"{Config.result_dir}/results_{year}_{suffix}.xlsx"
    df.to_excel(path, index=False)
    logger.info(f"结果已保存: {path}")


def plot_r2_bar(ids, y_true, y_pred, title, year):
    df = pd.DataFrame({'id': ids, 'true': y_true, 'pred': y_pred})
    buoys = sorted(list(set(ids)))
    r2_list = []

    for b in buoys:
        sub = df[df['id'] == b]
        if len(sub) > 5:
            r2_list.append(r2_score(sub['true'], sub['pred']))
        else:
            r2_list.append(0)

    plt.figure(figsize=(12, 5))
    plt.bar(buoys, r2_list, color='skyblue', edgecolor='k')
    plt.title(f"{year} {title} R2 Score by Buoy")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{Config.result_dir}/{year}_{title.replace(' ', '_')}_R2.png")
    plt.close()


def main():
    evaluate_test_set()
    predict_future_years()


if __name__ == "__main__":
    main()
