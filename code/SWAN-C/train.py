# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
import re
import os
import matplotlib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool
import random
from pathlib import Path

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset
from tqdm import tqdm
from datetime import datetime
import logging
import pickle

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('./results_multi_bias', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./results_multi_bias/train_swan_c.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()


class Config:
    """配置类"""

    _BASE_DIR = Path(__file__).resolve().parent.parent.parent   # repo root
    _DATA_DIR = _BASE_DIR / "data" / "uk"
    _WEIGHTS_DIR = _BASE_DIR / "weights"

    data_files_2020 = {
        "depth": str(_DATA_DIR / "swan_depth_2020.csv"),
        "wind": str(_DATA_DIR / "wind_wave_initial_data_2020.csv"),
        "bias": str(_DATA_DIR / "bias_data" / "bias_data_2020.csv")
    }
    data_files_2021 = {
        "depth": str(_DATA_DIR / "swan_depth_2021.csv"),
        "wind": str(_DATA_DIR / "wind_wave_initial_data_2021.csv"),
        "bias": str(_DATA_DIR / "bias_data" / "bias_data_2021.csv")
    }

    batch_size = 64
    lr = 0.03
    epochs = 15000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    result_dir = "./results_multi_bias"
    model_path = str(_WEIGHTS_DIR / "best_multi_bias_model.cbm")
    scaler_path = str(_WEIGHTS_DIR / "scaler_multi_bias.pkl")

    all_buoys = [
        "Bos_waves", "HgI_waves", "ChP_waves", "Mlf_waves", "BkB_waves", "Clv_waves", "Csl_waves", "Dwl_waves",
        "Flk_waves", "GdS_waves", "Hpg_waves", "LoB_waves", "Nbg_waves", "Plv_waves", "Pnz_waves", "Prp_waves",
        "RhF_waves", "Rst_waves", "SBy_waves", "Sca_waves", "StB_waves", "Wtb_waves", "Bdf_waves", "Hrn_waves",
        "Fxs_waves", "PBy_waves", "WBy_waves"
    ]

    coord_cols = ['depth']
    spatial_points = 1
    depth_indices = [200]
    grid_spacing = 50

    train_year = 2020
    train_months = list(range(1, 13))
    test_year = 2021
    test_months = list(range(1, 13))

    target_cols = ['bias_hs', 'bias_tm']


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

        raw_cols = ['swh', 'wind_speed', 'wind_direction', 'mwd', 'mwp', 'tide']
        for col in raw_cols:
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
        for target, scaler in zip(['bias_hs', 'bias_tm'], [self.bias_hs_scaler, self.bias_tm_scaler]):
            if target in df.columns:
                data = df[[target]].dropna()
                if not data.empty:
                    scaler.fit(data)

        if 'depth200' in df.columns:
            self.depth_scaler.fit(df[['depth200']].dropna())

        physical_features = ['swh', 'wind_speed', 'mwp', 'tide',
                             'sin_wd', 'cos_wd', 'sin_mwd', 'cos_mwd']

        for feat in physical_features:
            if feat in df.columns:
                valid_data = df[[feat]].dropna()
                if not valid_data.empty:
                    scaler = StandardScaler()
                    scaler.fit(valid_data)
                    self.feature_scalers[feat] = scaler

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


class MultiBiasDataset(Dataset):
    def __init__(self, year, months, buoy_ids=None, data_type='train'):
        self.year = year
        self.months = months
        self.buoy_ids = buoy_ids if buoy_ids is not None else Config.all_buoys
        self.data_type = data_type

        try:
            self._load_spatial_system()
            self.df = self._load_and_merge_data()
            if self.df.empty:
                raise ValueError("Merged dataset is empty")

            self.df = SimpleFeatureProcessor.add_basic_features(self.df)
            self._validate_dataset()
            logger.info(f"{data_type} Set ({year}): {len(self)} samples loaded.")
        except Exception as e:
            logger.error(f"Dataset load error: {e}")
            raise

    def __len__(self):
        return len(self.df)

    def _load_spatial_system(self):
        path = Config.data_files_2020["depth"] if self.year == 2020 else Config.data_files_2021["depth"]
        depth_df = pd.read_csv(path)
        depth_df = depth_df[depth_df['id'].isin(self.buoy_ids)]
        self.coord_system = SpatialCoordManager.load_coordinates(depth_df)

    def _load_and_merge_data(self) -> pd.DataFrame:
        files = Config.data_files_2020 if self.year == 2020 else Config.data_files_2021
        data_frames = {}

        for name, path in files.items():
            if not os.path.exists(path):
                logger.warning(f"文件 {path} 不存在")
                return pd.DataFrame()

            try:
                df = pd.read_csv(path, dtype={'time': str})
                t_formatted = pd.to_datetime(df['time'], format='%Y%m%d%H', errors='coerce')
                if t_formatted.isna().any():
                    mask = t_formatted.isna()
                    t_auto = pd.to_datetime(df.loc[mask, 'time'], errors='coerce')
                    t_formatted.loc[mask] = t_auto
                df['time'] = t_formatted
                df.dropna(subset=['time'], inplace=True)
                df['time'] = df['time'].dt.floor('h')
                df = df[df['id'].isin(self.buoy_ids)]
                df = df[df['time'].dt.month.isin(self.months)]
                df.drop_duplicates(subset=['id', 'time'], keep='first', inplace=True)
                data_frames[name] = df
            except Exception as e:
                logger.error(f"读取 {name} 失败: {e}")
                return pd.DataFrame()

        if len(data_frames) < 3:
            return pd.DataFrame()

        merged = data_frames['depth'].merge(data_frames['wind'], on=['id', 'time'], how='inner', suffixes=('', '_drop'))
        merged = merged.loc[:, ~merged.columns.str.contains('_drop')]
        merged = merged.merge(data_frames['bias'], on=['id', 'time'], how='inner', suffixes=('', '_drop'))
        merged = merged.loc[:, ~merged.columns.str.contains('_drop')]
        merged.dropna(subset=['bias_hs', 'bias_tm'], inplace=True)

        return merged

    def _validate_dataset(self):
        req_cols = ['id', 'time', 'swh', 'bias_hs', 'bias_tm', 'depth200', 'sin_wd', 'cos_wd']
        missing = [c for c in req_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

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


def prepare_catboost_data(dataset):
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


def plot_feature_importance(model, feature_names):
    importance = model.get_feature_importance()
    indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance for Bias Prediction")
    plt.bar(range(len(importance)), importance[indices], align="center")
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig(f"{Config.result_dir}/feature_importance.png")
    plt.close()

    logger.info("--- Feature Importance ---")
    for i in indices:
        logger.info(f"{feature_names[i]}: {importance[i]:.4f}")


def train_multi_output():
    os.makedirs(Config.result_dir, exist_ok=True)

    if not os.path.exists(Config.data_files_2020['bias']):
        logger.error(f"找不到 2020 Bias 文件: {Config.data_files_2020['bias']}")
        return None, None

    train_dataset = MultiBiasDataset(Config.train_year, Config.train_months, data_type='train')
    train_df_raw = train_dataset.df.sort_values('time').reset_index(drop=True)
    train_idx, val_idx = train_test_split(np.arange(len(train_df_raw)), test_size=0.2, random_state=42, shuffle=True)
    train_subset = train_dataset.df.iloc[train_idx]

    scaler = HierarchicalScaler()
    scaler.fit(train_subset)
    train_dataset.df = scaler.transform(train_dataset.df)

    with open(Config.scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    X_all, y_all, _, _ = prepare_catboost_data(train_dataset)
    X_train, y_train = X_all.iloc[train_idx], y_all[train_idx]
    X_val, y_val = X_all.iloc[val_idx], y_all[val_idx]

    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)

    logger.info("初始化 CatBoostRegressor (MultiRMSE)...")
    model = CatBoostRegressor(
        iterations=Config.epochs,
        learning_rate=Config.lr,
        depth=6,
        loss_function='MultiRMSE',
        eval_metric='MultiRMSE',
        verbose=500,
        random_seed=42,
        early_stopping_rounds=500,
        task_type='GPU' if torch.cuda.is_available() else 'CPU'
    )

    model.fit(train_pool, eval_set=val_pool)
    model.save_model(Config.model_path)

    plot_feature_importance(model, X_train.columns.tolist())

    return model, scaler


def evaluate_and_test():
    if not os.path.exists(Config.model_path):
        logger.warning("模型文件不存在，跳过评估。")
        return

    model = CatBoostRegressor()
    model.load_model(Config.model_path)

    with open(Config.scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    if not os.path.exists(Config.data_files_2021['bias']):
        logger.error("找不到 2021 Bias 文件，请先生成。")
        return

    test_dataset = MultiBiasDataset(Config.test_year, Config.test_months, data_type='test')
    test_dataset.df = scaler.transform(test_dataset.df)

    X_test, y_test_scaled, ids, times = prepare_catboost_data(test_dataset)
    preds_scaled = model.predict(X_test)

    pred_hs_scaled = preds_scaled[:, 0]
    pred_tm_scaled = preds_scaled[:, 1]
    true_hs_scaled = y_test_scaled[:, 0]
    true_tm_scaled = y_test_scaled[:, 1]

    pred_hs, pred_tm = scaler.inverse_transform_targets(pred_hs_scaled, pred_tm_scaled)
    true_hs, true_tm = scaler.inverse_transform_targets(true_hs_scaled, true_tm_scaled)

    logger.info("评估结果 (Hs):")
    calculate_metrics(true_hs, pred_hs, "Hs")
    logger.info("评估结果 (Tm):")
    calculate_metrics(true_tm, pred_tm, "Tm")

    save_results_excel(ids, times, true_hs, pred_hs, true_tm, pred_tm)
    plot_r2_by_buoy(ids, true_hs, pred_hs, "Hs_Bias")
    plot_r2_by_buoy(ids, true_tm, pred_tm, "Tm_Bias")


def calculate_metrics(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    logger.info(f"[{name}] RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


def save_results_excel(ids, times, true_hs, pred_hs, true_tm, pred_tm):
    df = pd.DataFrame({
        'Buoy_ID': ids, 'Time': times,
        'True_Bias_Hs': true_hs, 'Pred_Bias_Hs': pred_hs,
        'True_Bias_Tm': true_tm, 'Pred_Bias_Tm': pred_tm
    })
    path = f"{Config.result_dir}/final_test_results.xlsx"
    df.to_excel(path, index=False)
    logger.info(f"Saved results to {path}")


def plot_r2_by_buoy(ids, y_true, y_pred, title_prefix):
    df = pd.DataFrame({'id': ids, 'true': y_true, 'pred': y_pred})
    r2_scores = []
    buoys = sorted(list(set(ids)))
    for b in buoys:
        sub = df[df['id'] == b]
        if len(sub) > 5:
            r2_scores.append(r2_score(sub['true'], sub['pred']))
        else:
            r2_scores.append(0)
    plt.figure(figsize=(15, 6))
    plt.bar(buoys, r2_scores, color='skyblue', edgecolor='black')
    plt.title(f'{title_prefix} - R2 Score by Buoy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{Config.result_dir}/r2_by_buoy_{title_prefix}.png")
    plt.close()


def main():
    logger.info("启动 SWAN-C 双目标偏差预测训练 (Hs & Tm)...")
    train_multi_output()
    evaluate_and_test()
    logger.info("训练完成。")


if __name__ == "__main__":
    main()
