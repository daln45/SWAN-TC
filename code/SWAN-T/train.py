# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
import os
import matplotlib
from sklearn.preprocessing import StandardScaler, MinMaxScaler

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
import pickle
import random
import math
from pathlib import Path

# ==========================================
# 全局设置
# ==========================================

# 设置随机种子以确保可重复性
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 设置中文字体并解决负号问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'STIXGeneral']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./train_swan_t.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()


# ==========================================
# 配置类
# ==========================================
class DataEfficiencyConfig:
    """数据效率实验配置类 - 多任务版 (Hs + Tm)"""

    # ========== 1. 实验配置 ==========
    experiments = {
        # 组1：混合虚拟数据 (use_virtual=True)
        'exp_1_pure_virtual': {'ratio': 0.0, 'use_virtual': True, 'name': '0% Real + Virtual'},
        'exp_2_25_percent': {'ratio': 0.25, 'use_virtual': True, 'name': '25% Real + Virtual'},
        'exp_3_50_percent': {'ratio': 0.50, 'use_virtual': True, 'name': '50% Real + Virtual'},
        'exp_4_75_percent': {'ratio': 0.75, 'use_virtual': True, 'name': '75% Real + Virtual'},
        'exp_5_100_percent': {'ratio': 1.0, 'use_virtual': True, 'name': '100% Real + Virtual'},

        # 组2：仅真实数据 (use_virtual=False)
        'exp_6_50_real_only': {'ratio': 0.50, 'use_virtual': False, 'name': '50% Real Only'},
        'exp_7_100_real_only': {'ratio': 1.0, 'use_virtual': False, 'name': '100% Real Only'}
    }

    # 训练月份固定为全年
    train_months = list(range(1, 13))

    # ========== 2. 文件路径配置 (仅保留2020) ==========
    _BASE_DIR = Path(__file__).resolve().parent.parent.parent   # repo root
    _DATA_DIR = _BASE_DIR / "data" / "uk"

    data_files_2020 = {
        "depth": str(_DATA_DIR / "swan_depth_2020.csv"),
        "wind": str(_DATA_DIR / "wind_wave_initial_data_2020.csv"),
        "swan_hs": str(_DATA_DIR / "swan_hs_2020.csv"),
        "swan_tm": str(_DATA_DIR / "swan_tm_2020.csv")
    }

    # 虚拟数据文件配置 (合并加载)
    data_files_virtual = {
        "depth": str(_DATA_DIR / "virtual_profile_depth.csv"),
        "hs": str(_DATA_DIR / "virtual_profile_hs.csv"),
        "tm": str(_DATA_DIR / "virtual_profile_tm.csv"),
        "wind": str(_DATA_DIR / "boundary_data_2020.csv")
    }

    # ========== 3. 训练与测试时间配置 (统一为2020) ==========
    train_year = 2020
    test_year = 2020
    test_months = list(range(1, 13))

    # 浮标配置
    all_real_buoys = ["Bos_waves", "HgI_waves", "ChP_waves", "Mlf_waves",
                      "BkB_waves", "Clv_waves", "Csl_waves", "Dwl_waves",
                      "Flk_waves", "GdS_waves", "Hpg_waves",
                      "LoB_waves", "Nbg_waves", "Plv_waves", "Pnz_waves",
                      "Prp_waves", "RhF_waves", "Rst_waves",
                      "SBy_waves", "Sca_waves", "StB_waves", "Wtb_waves", "Bdf_waves"]

    test_buoys = ["Hrn_waves", "Fxs_waves", "PBy_waves", "WBy_waves"]
    val_buoys = ["BkB_waves", "SBy_waves", "Bdf_waves", "Flk_waves", "Csl_waves"]

    # 虚拟浮标ID列表
    virtual_original_buoys = [f"output_case_{i:05d}_virtual_{j:02d}" for i in range(1, 2001) for j in range(50)]
    virtual_dune_buoys = [f"output_case_{i:05d}_dune_{j:02d}" for i in range(1, 2001) for j in range(50)]

    @classmethod
    def get_train_only_buoys(cls):
        return [buoy for buoy in cls.all_real_buoys
                if buoy not in cls.test_buoys and buoy not in cls.val_buoys]

    @classmethod
    def get_all_training_buoys(cls):
        return (cls.get_train_only_buoys() + cls.virtual_original_buoys + cls.virtual_dune_buoys)

    # Transformer架构配置
    d_model = 256
    nhead = 8
    num_layers = 6
    dim_feedforward = 512
    dropout = 0.1
    activation = 'gelu'

    # 输入特征配置
    input_channels = {
        'spatial': 1,
        'physical': 7,
        'output': 2  # Hs + Tm
    }

    # 训练配置
    batch_size = 64
    lr = 0.0001
    epochs = 200
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 空间网格配置
    coord_cols = ['depth']
    spatial_points = 64
    depth_indices = list(range(10, 200, 3))
    grid_spacing = 50
    smooth_lambda = 0.0

    # 结果保存路径
    @classmethod
    def get_paths(cls, experiment_id):
        base_dir = Path(f"./results_data_efficiency_{experiment_id}")
        return {
            'result_dir': base_dir,
            'train_test_result_dir': base_dir / "train_test_results",
            'model_path': Path(f"./best_model_{experiment_id}.pth"),
            'scaler_path': base_dir / f"scaler_{experiment_id}.pkl"
        }


# ==========================================
# 数据处理与Dataset
# ==========================================
class SpatialCoordManager:
    """空间坐标管理器"""

    @classmethod
    def load_coordinates(cls, depth_df: pd.DataFrame, months_filter=None) -> Dict[str, Dict[str, np.ndarray]]:
        depth_map = {}
        pattern = re.compile(r'^(depth)(\d{3})$')

        if 'id' not in depth_df.columns or 'time' not in depth_df.columns:
            if depth_df.empty: return {'x': [], 'y': [], 'depth_map': {}}
            raise ValueError("深度数据缺少 'id' 或 'time' 列")

        depth_df['time'] = pd.to_datetime(depth_df['time'], format='%Y%m%d%H', errors='coerce').dt.floor('h')
        depth_df = depth_df.dropna(subset=['time'])

        if months_filter:
            depth_df = depth_df[depth_df['time'].dt.month.isin(months_filter)]

        coord_map = {}
        for col in depth_df.columns:
            if col in ['id', 'time']: continue
            match = pattern.match(col)
            if match:
                var_type, idx = match.groups()
                idx_num = int(idx)
                if var_type in DataEfficiencyConfig.coord_cols and idx_num in DataEfficiencyConfig.depth_indices:
                    coord_map[idx] = {var_type: col}

        if not coord_map: return {'x': [], 'y': [], 'depth_map': {}}

        for buoy_id in depth_df['id'].unique():
            buoy_df = depth_df[depth_df['id'] == buoy_id]
            depth_map[buoy_id] = {}
            for _, row in buoy_df.iterrows():
                depth_values = []
                for idx_num in DataEfficiencyConfig.depth_indices:
                    idx_str = f"{idx_num:03d}"
                    val = row[coord_map[idx_str]['depth']] if idx_str in coord_map else np.nan
                    depth_values.append(val)
                depth_map[buoy_id][row['time']] = np.array(depth_values, dtype=np.float32)

        x = DataEfficiencyConfig.grid_spacing * np.array(DataEfficiencyConfig.depth_indices, dtype=np.float32)
        y = np.zeros(DataEfficiencyConfig.spatial_points, dtype=np.float32)
        return {'x': x, 'y': y, 'depth_map': depth_map}


class SimpleFeatureProcessor:
    @staticmethod
    def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        df = df.copy()
        required_cols = ['wind_direction', 'mwd', 'wind_speed', 'mwp', 'swh']
        missing = [c for c in required_cols if c not in df.columns]
        if missing: raise ValueError(f"缺少列: {missing}")

        if 'alpc' not in df.columns: df['alpc'] = 0.0
        if 'tide' not in df.columns: df['tide'] = 0.0

        feature_cols = ['swh', 'wind_speed', 'wind_direction', 'mwd', 'mwp', 'alpc', 'tide']
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
        return df


class ScenarioScaler:
    """标准化器 (Hs + Tm)"""

    def __init__(self):
        self.hs_scaler = StandardScaler()
        self.tm_scaler = StandardScaler()
        self.depth_scaler = StandardScaler()
        self.scenario_scalers = {}
        self.is_fitted = False

    def fit_with_data(self, df: pd.DataFrame):
        """使用传入的数据集进行训练"""
        logger.info(f"开始训练标准化器 (样本数: {len(df)})...")
        if df.empty:
            raise ValueError("用于标准化的数据集为空，无法继续！")

        # 1. Hs
        hs_cols = [f'hs{idx:03d}' for idx in DataEfficiencyConfig.depth_indices]
        valid_hs = [c for c in hs_cols if c in df.columns]
        if valid_hs: self.hs_scaler.fit(df[valid_hs].dropna().astype(np.float32))

        # 2. Tm
        tm_cols = [f'tm{idx:03d}' for idx in DataEfficiencyConfig.depth_indices]
        valid_tm = [c for c in tm_cols if c in df.columns]
        if valid_tm: self.tm_scaler.fit(df[valid_tm].dropna().astype(np.float32))

        # 3. Depth
        depth_cols = [f'depth{idx:03d}' for idx in DataEfficiencyConfig.depth_indices]
        valid_depth = [c for c in depth_cols if c in df.columns]
        if valid_depth: self.depth_scaler.fit(df[valid_depth].dropna().astype(np.float32))

        # 4. Features
        features = ['swh', 'wind_speed', 'wind_direction', 'mwd', 'mwp', 'alpc', 'tide']
        for feat in features:
            if feat in ['wind_direction', 'mwd', 'alpc', 'swh', 'wind_speed', 'mwp', 'tide']:
                scaler = MinMaxScaler(feature_range=(-1, 1))
            else:
                scaler = StandardScaler()

            if feat in df.columns:
                data = df[[feat]].dropna().astype(np.float32)
                if not data.empty:
                    scaler.fit(data)
                    self.scenario_scalers[feat] = scaler

        self.is_fitted = True
        logger.info("标准化器训练完成")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted: raise ValueError("Scaler未训练")
        if df.empty: return df
        df = df.copy()

        # Hs
        hs_cols = [f'hs{idx:03d}' for idx in DataEfficiencyConfig.depth_indices]
        valid_hs = [c for c in hs_cols if c in df.columns]
        if valid_hs and hasattr(self.hs_scaler, 'mean_'):
            df[valid_hs] = self.hs_scaler.transform(df[valid_hs].fillna(0).astype(np.float32))

        # Tm
        tm_cols = [f'tm{idx:03d}' for idx in DataEfficiencyConfig.depth_indices]
        valid_tm = [c for c in tm_cols if c in df.columns]
        if valid_tm and hasattr(self.tm_scaler, 'mean_'):
            df[valid_tm] = self.tm_scaler.transform(df[valid_tm].fillna(0).astype(np.float32))

        # Depth
        depth_cols = [f'depth{idx:03d}' for idx in DataEfficiencyConfig.depth_indices]
        valid_depth = [c for c in depth_cols if c in df.columns]
        if valid_depth and hasattr(self.depth_scaler, 'mean_'):
            df[valid_depth] = self.depth_scaler.transform(df[valid_depth].fillna(0).astype(np.float32))

        # Features
        for feat, scaler in self.scenario_scalers.items():
            if feat in df.columns:
                df[feat] = scaler.transform(df[[feat]].fillna(0).astype(np.float32))

        return df

    def inverse_transform_targets(self, targets: np.ndarray, var_type: str) -> np.ndarray:
        if var_type == 'hs' and hasattr(self.hs_scaler, 'mean_'):
            return self.hs_scaler.inverse_transform(targets)
        elif var_type == 'tm' and hasattr(self.tm_scaler, 'mean_'):
            return self.tm_scaler.inverse_transform(targets)
        return targets


class DualVirtualDataset(Dataset):
    """
    数据集类
    - 支持混合比例采样 (real_data_ratio)
    - 支持虚拟数据开关 (include_virtual)
    - 虚拟数据合并加载
    - 统一使用 2020 数据
    """

    def __init__(self, months, year, buoy_ids, data_type='train', real_data_ratio=1.0, include_virtual=True):
        self.months = months
        self.year = year
        self.buoy_ids = buoy_ids
        self.data_type = data_type
        self.real_data_ratio = real_data_ratio
        self.include_virtual = include_virtual

        self._load_spatial_system()
        self.df = self._load_and_merge_data()

        if not self.df.empty:
            self.df = SimpleFeatureProcessor.add_basic_features(self.df)

        logger.info(
            f"[{data_type}] 数据集构建完成 (Ratio={real_data_ratio}, Virtual={include_virtual}): 总样本数={len(self)}")

    def __len__(self):
        return len(self.df)

    def _load_spatial_system(self):
        depth_dfs = []
        # 1. Real Data (统一使用 2020 文件)
        d_file = DataEfficiencyConfig.data_files_2020["depth"]
        if os.path.exists(d_file):
            df = pd.read_csv(d_file)
            target_ids = [b for b in self.buoy_ids if not b.startswith('output')]
            df = df[df['id'].isin(target_ids)]
            depth_dfs.append(df)

        # 2. Virtual Data
        if self.year == DataEfficiencyConfig.train_year and self.include_virtual:
            if os.path.exists(DataEfficiencyConfig.data_files_virtual["depth"]):
                depth_dfs.append(pd.read_csv(DataEfficiencyConfig.data_files_virtual["depth"]))

        if depth_dfs:
            full_depth = pd.concat(depth_dfs, ignore_index=True)
            self.coord_system = SpatialCoordManager.load_coordinates(full_depth, self.months)
        else:
            self.coord_system = {'depth_map': {}}

    def _load_and_merge_data(self) -> pd.DataFrame:
        all_data = []

        # 1. Real Data
        real = self._load_real_data()
        if not real.empty and self.data_type == 'train':
            if self.real_data_ratio == 0.0:
                logger.info(f"纯虚拟模式，移除真实数据")
                real = pd.DataFrame(columns=real.columns)
            elif self.real_data_ratio < 1.0:
                logger.info(f"真实数据采样: {self.real_data_ratio}")
                real = real.sample(frac=self.real_data_ratio, random_state=42)
        if not real.empty:
            all_data.append(real)

        # 2. Virtual Data
        if self.year == DataEfficiencyConfig.train_year and self.include_virtual:
            virt = self._load_virtual_data(DataEfficiencyConfig.data_files_virtual)
            if not virt.empty:
                all_data.append(virt)

        if all_data:
            merged = pd.concat(all_data, ignore_index=True)
            return merged.loc[:, ~merged.columns.str.contains('_DROP')]
        return pd.DataFrame()

    def _load_real_data(self) -> pd.DataFrame:
        files = DataEfficiencyConfig.data_files_2020
        dfs = {}
        for k in ['depth', 'wind', 'swan_hs', 'swan_tm']:
            if k in files and os.path.exists(files[k]):
                df = pd.read_csv(files[k])
                df = df[df['id'].isin(self.buoy_ids)]
                df['time'] = pd.to_datetime(df['time'], format='%Y%m%d%H', errors='coerce').dt.floor('h')
                df = df[df['time'].dt.month.isin(self.months)]
                dfs[k] = df
            else:
                return pd.DataFrame()

        merged = dfs['depth'].merge(dfs['wind'], on=['id', 'time'], how='inner') \
            .merge(dfs['swan_hs'], on=['id', 'time'], how='inner') \
            .merge(dfs['swan_tm'], on=['id', 'time'], how='inner')
        return merged

    def _load_virtual_data(self, file_config) -> pd.DataFrame:
        target_ids = [b for b in self.buoy_ids if b.startswith('output')]
        if not target_ids: return pd.DataFrame()

        dfs = {}
        # 1. 加载 HS, TM, DEPTH
        for k in ['hs', 'tm', 'depth']:
            if k in file_config and os.path.exists(file_config[k]):
                df = pd.read_csv(file_config[k])
                df = df[df['id'].isin(target_ids)]
                df['time'] = pd.to_datetime(df['time'], format='%Y%m%d%H', errors='coerce').dt.floor('h')
                df = df[df['time'].dt.month.isin(self.months)]
                dfs[k] = df
            else:
                return pd.DataFrame()

        # 2. 加载 WIND 并合并
        if 'wind' in file_config and os.path.exists(file_config['wind']):
            wind_df = pd.read_csv(file_config['wind'])

            # 自动识别时间列 (time 或 datetime)
            t_col = 'time' if 'time' in wind_df.columns else 'datetime'
            wind_df['time'] = pd.to_datetime(wind_df[t_col], errors='coerce').dt.floor('h')

            # --- A. 准备波浪数据 (左表) ---
            merged_temp = dfs['hs'].merge(dfs['tm'], on=['id', 'time'], how='inner') \
                .merge(dfs['depth'], on=['id', 'time'], how='inner')

            # 从 output_case_00001... 中提取 00001
            merged_temp['scene_num'] = merged_temp['id'].str.extract(r'output_case_(\d{5})')[0]
            # 构造连接键 synthetic_00001
            merged_temp['scene_id'] = 'synthetic_' + merged_temp['scene_num']

            # --- B. 准备风场数据 (右表) ---
            # 逻辑：无论是有 sample_rank 还是 id，都统一构造成 synthetic_XXXXX
            if 'sample_rank' in wind_df.columns:
                # 适配 boundary_data_2020.csv
                wind_df['scene_num_temp'] = wind_df['sample_rank'].astype(str).str.zfill(5)
                wind_df['scene_id'] = 'synthetic_' + wind_df['scene_num_temp']
            elif 'id' in wind_df.columns:
                # 适配旧版 synthetic_wave_params.csv
                wind_df['scene_num_temp'] = wind_df['id'].astype(str).str.extract(r'(\d+)')[0].str.zfill(5)
                wind_df['scene_id'] = 'synthetic_' + wind_df['scene_num_temp']

            # --- C. 关键修复：筛选列以避免冲突 ---
            # 我们只保留 wind_df 中的 特征列 + 连接键
            # 这样就不会把右表的 scene_num_temp 或其他列带进去导致后缀冲突
            feature_cols = ['wind_speed', 'wind_direction', 'mwd', 'mwp', 'swh', 'alpc', 'tide']
            cols_to_keep = ['scene_id', 'time'] + [c for c in feature_cols if c in wind_df.columns]

            wind_df_clean = wind_df[cols_to_keep]

            # --- D. 合并 ---
            final_merged = merged_temp.merge(wind_df_clean, on=['scene_id', 'time'], how='inner')

            # 现在可以安全删除左表生成的辅助列了
            return final_merged.drop(columns=['scene_id', 'scene_num'])

        return pd.DataFrame()

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        buoy_id = row['id']
        ts = row['time']

        if buoy_id not in self.coord_system['depth_map'] or ts not in self.coord_system['depth_map'][buoy_id]:
            depth = np.zeros(DataEfficiencyConfig.spatial_points, dtype=np.float32)
        else:
            depth = self.coord_system['depth_map'][buoy_id][ts].astype(np.float32)

        spatial_input = depth.reshape(1, -1)

        feats = np.array([
            row['swh'], row['wind_speed'], row['wind_direction'],
            row['mwd'], row['mwp'], row['alpc'], row['tide']
        ], dtype=np.float32)
        physical_input = np.tile(feats, (DataEfficiencyConfig.spatial_points, 1)).T

        hs_cols = [f'hs{idx:03d}' for idx in DataEfficiencyConfig.depth_indices]
        tm_cols = [f'tm{idx:03d}' for idx in DataEfficiencyConfig.depth_indices]

        target = np.full((DataEfficiencyConfig.spatial_points, 2), np.nan, dtype=np.float32)

        vals_h = row[hs_cols].values.astype(np.float32)
        target[:len(vals_h), 0] = vals_h

        vals_t = row[tm_cols].values.astype(np.float32)
        target[:len(vals_t), 1] = vals_t

        spatial_mask = ~np.isnan(spatial_input)
        physical_mask = ~np.isnan(feats)
        target_mask = ~np.isnan(target)

        return (
            torch.FloatTensor(spatial_input),
            torch.FloatTensor(physical_input),
            torch.FloatTensor(target),
            torch.BoolTensor(spatial_mask),
            torch.BoolTensor(physical_mask),
            torch.BoolTensor(target_mask),
            buoy_id,
            ts.strftime('%Y%m%d%H')
        )


# ==========================================
# 模型架构 (Transformer)
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=DataEfficiencyConfig.dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class FeatureEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.spatial_projection = nn.Linear(DataEfficiencyConfig.input_channels['spatial'], d_model)
        self.physical_projection = nn.Linear(DataEfficiencyConfig.input_channels['physical'], d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(DataEfficiencyConfig.dropout)
        self.modal_weights = nn.Parameter(torch.ones(2))

    def forward(self, spatial, physical, spatial_mask, physical_mask):
        spatial = spatial.transpose(1, 2)
        physical = physical.transpose(1, 2)
        spatial_emb = self.spatial_projection(spatial)
        physical_emb = self.physical_projection(physical)
        weights = torch.softmax(self.modal_weights, dim=0)
        combined = (weights[0] * spatial_emb + weights[1] * physical_emb)
        combined = self.dropout(self.layer_norm(combined))
        mask = spatial_mask.squeeze(1) & (
            physical_mask.all(dim=1, keepdim=True).expand(-1,
                                                          DataEfficiencyConfig.spatial_points) if physical_mask.dim() == 2 else physical_mask)
        return combined, mask


class TransformerWavePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = DataEfficiencyConfig.d_model
        self.feature_embedding = FeatureEmbedding(self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=DataEfficiencyConfig.spatial_points)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=DataEfficiencyConfig.nhead,
                                                   dim_feedforward=DataEfficiencyConfig.dim_feedforward,
                                                   dropout=DataEfficiencyConfig.dropout,
                                                   activation=DataEfficiencyConfig.activation,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=DataEfficiencyConfig.num_layers)
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2), nn.GELU(), nn.Dropout(DataEfficiencyConfig.dropout),
            nn.Linear(self.d_model // 2, self.d_model // 4), nn.GELU(), nn.Dropout(DataEfficiencyConfig.dropout),
            nn.Linear(self.d_model // 4, DataEfficiencyConfig.input_channels['output'])
        )

    def forward(self, spatial, physical, spatial_mask, physical_mask):
        emb, mask = self.feature_embedding(spatial, physical, spatial_mask, physical_mask)
        emb = self.pos_encoder(emb.transpose(0, 1)).transpose(0, 1)
        out = self.transformer_encoder(emb, src_key_padding_mask=~mask)
        pred = self.output_projection(out)
        return pred.permute(0, 2, 1)


class UncertaintyLoss(nn.Module):
    def __init__(self, smooth_lambda=0.0):
        super().__init__()
        self.smooth_lambda = smooth_lambda
        self.log_vars = nn.Parameter(torch.zeros(2))

    def masked_mse(self, pred, target, mask):
        diff = (pred - target) ** 2
        diff = diff * mask.float()
        valid = mask.sum().clamp(min=1).float()
        return diff.sum() / valid

    def forward(self, predictions, targets, target_mask):
        targets = targets.permute(0, 2, 1)
        target_mask = target_mask.permute(0, 2, 1)
        pred_h, pred_t = predictions[:, 0, :], predictions[:, 1, :]
        target_h, target_t = targets[:, 0, :], targets[:, 1, :]
        mask_h, mask_t = target_mask[:, 0, :], target_mask[:, 1, :]

        loss_h = self.masked_mse(pred_h, target_h, mask_h)
        loss_t = self.masked_mse(pred_t, target_t, mask_t)

        if self.smooth_lambda > 0:
            loss_h += self.smooth_lambda * torch.mean((pred_h[:, 1:] - pred_h[:, :-1]) ** 2)
            loss_t += self.smooth_lambda * torch.mean((pred_t[:, 1:] - pred_t[:, :-1]) ** 2)

        s_h, s_t = self.log_vars[0], self.log_vars[1]
        weighted_loss_h = 0.5 * torch.exp(-s_h) * loss_h + 0.5 * s_h
        weighted_loss_t = 0.5 * torch.exp(-s_t) * loss_t + 0.5 * s_t

        return weighted_loss_h + weighted_loss_t, loss_h.item(), loss_t.item(), s_h.item(), s_t.item()


# ==========================================
# 训练与评估流程
# ==========================================
def evaluate_transformer(model, loader, criterion, device, scaler):
    model.eval()
    losses = []
    res = {'id': [], 'time': [], 'pred_h': [], 'targ_h': [], 'mask_h': [], 'pred_tm': [], 'targ_tm': [], 'mask_tm': []}

    with torch.no_grad():
        for batch in loader:
            spatial, physical, targets, s_mask, p_mask, t_mask, ids, times = batch
            spatial, physical = spatial.to(device), physical.to(device)
            targets = targets.to(device)
            s_mask, p_mask, t_mask = s_mask.to(device), p_mask.to(device), t_mask.to(device)

            outputs = model(spatial, physical, s_mask, p_mask)
            loss, _, _, _, _ = criterion(outputs, targets, t_mask)
            losses.append(loss.item())

            outputs = outputs.cpu().numpy()
            targets = targets.cpu().numpy()
            t_mask = t_mask.cpu().numpy()

            pred_h, pred_t = outputs[:, 0, :], outputs[:, 1, :]
            targ_h, targ_t = targets[:, :, 0], targets[:, :, 1]
            mask_h, mask_t = t_mask[:, :, 0], t_mask[:, :, 1]

            res['id'].extend(ids)
            res['time'].extend(times)
            res['pred_h'].append(scaler.inverse_transform_targets(pred_h, 'hs'))
            res['targ_h'].append(scaler.inverse_transform_targets(targ_h, 'hs'))
            res['mask_h'].append(mask_h)
            res['pred_tm'].append(scaler.inverse_transform_targets(pred_t, 'tm'))
            res['targ_tm'].append(scaler.inverse_transform_targets(targ_t, 'tm'))
            res['mask_tm'].append(mask_t)

    for k in ['pred_h', 'targ_h', 'mask_h', 'pred_tm', 'targ_tm', 'mask_tm']:
        res[k] = np.concatenate(res[k], axis=0) if res[k] else np.array([])
    return np.mean(losses) if losses else 0.0, res


def train_scaling_experiment(experiment_id, experiment_config):
    logger.info("=" * 60)
    logger.info(f"开始实验: {experiment_config['name']} (Ratio: {experiment_config['ratio']})")
    logger.info("=" * 60)

    paths = DataEfficiencyConfig.get_paths(experiment_id)
    paths['result_dir'].mkdir(parents=True, exist_ok=True)
    paths['train_test_result_dir'].mkdir(parents=True, exist_ok=True)

    use_virtual = experiment_config.get('use_virtual', True)
    if use_virtual:
        target_buoy_ids = DataEfficiencyConfig.get_all_training_buoys()
    else:
        target_buoy_ids = DataEfficiencyConfig.get_train_only_buoys()

    # 1. 准备训练数据
    train_dataset = DualVirtualDataset(
        months=DataEfficiencyConfig.train_months,
        year=DataEfficiencyConfig.train_year,
        buoy_ids=target_buoy_ids,
        data_type='train',
        real_data_ratio=experiment_config['ratio'],
        include_virtual=use_virtual
    )

    if train_dataset.df.empty:
        logger.error(f"实验 {experiment_id} 训练集为空，跳过")
        return None, None, None

    # 2. 训练Scaler
    scaler = ScenarioScaler()
    scaler.fit_with_data(train_dataset.df)
    train_dataset.df = scaler.transform(train_dataset.df)

    # 3. 准备验证数据
    val_dataset = DualVirtualDataset(
        months=DataEfficiencyConfig.train_months,
        year=DataEfficiencyConfig.train_year,
        buoy_ids=DataEfficiencyConfig.val_buoys,
        data_type='val',
        real_data_ratio=1.0,
        include_virtual=False
    )
    val_dataset.df = scaler.transform(val_dataset.df)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=DataEfficiencyConfig.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=DataEfficiencyConfig.batch_size, shuffle=False, num_workers=4)

    logger.info(f"训练集样本: {len(train_dataset)} | 验证集样本: {len(val_dataset)}")

    # 模型初始化
    model = TransformerWavePredictor().to(DataEfficiencyConfig.device)
    criterion = UncertaintyLoss(smooth_lambda=DataEfficiencyConfig.smooth_lambda).to(DataEfficiencyConfig.device)
    optimizer = torch.optim.AdamW([
        {'params': model.parameters()},
        {'params': criterion.parameters(), 'lr': 1e-3}
    ], lr=DataEfficiencyConfig.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, min_lr=1e-6, patience=10)

    best_loss = float('inf')
    patience = 25
    epochs_no_improve = 0

    for epoch in range(DataEfficiencyConfig.epochs):
        model.train()
        epoch_loss = 0

        # tqdm 进度条
        for batch in tqdm(train_loader, desc=f"Exp {experiment_id} Epoch {epoch + 1}"):
            spatial, physical, targets, s_mask, p_mask, t_mask, _, _ = batch
            spatial, physical = spatial.to(DataEfficiencyConfig.device), physical.to(DataEfficiencyConfig.device)
            targets = targets.to(DataEfficiencyConfig.device)
            s_mask, p_mask, t_mask = s_mask.to(DataEfficiencyConfig.device), p_mask.to(
                DataEfficiencyConfig.device), t_mask.to(DataEfficiencyConfig.device)

            optimizer.zero_grad()
            outputs = model(spatial, physical, s_mask, p_mask)
            loss, _, _, _, _ = criterion(outputs, targets, t_mask)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # 计算验证集 Loss
        val_loss, _ = evaluate_transformer(model, val_loader, criterion, DataEfficiencyConfig.device, scaler)

        avg_train_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), paths['model_path'])
            logger.info(f"--> 保存最佳模型 (Best Val: {best_loss:.4f})")
        else:
            epochs_no_improve += 1
            logger.info(f"--> 验证集未提升 ({epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                logger.info(f"早停触发！最佳 Loss: {best_loss:.4f}")
                break

    with open(paths['scaler_path'], 'wb') as f:
        pickle.dump(scaler, f)
    return model, paths, scaler


def test_experiment(experiment_id, model, paths, scaler):
    logger.info(f"开始测试: {experiment_id}")

    test_dataset = DualVirtualDataset(
        months=DataEfficiencyConfig.test_months,
        year=DataEfficiencyConfig.test_year,
        buoy_ids=DataEfficiencyConfig.test_buoys,
        data_type='test',
        real_data_ratio=1.0,
        include_virtual=False
    )
    test_dataset.df = scaler.transform(test_dataset.df)
    test_loader = DataLoader(test_dataset, batch_size=DataEfficiencyConfig.batch_size, shuffle=False)
    criterion = UncertaintyLoss().to(DataEfficiencyConfig.device)

    loss, res = evaluate_transformer(model, test_loader, criterion, DataEfficiencyConfig.device, scaler)

    metrics = {}
    # h for Hs, tm for Tm
    for var, key in [('h', 'h'), ('t', 'tm')]:
        valid = res[f'mask_{key}'].astype(bool)
        if valid.sum() > 0:
            p, t = res[f'pred_{key}'][valid], res[f'targ_{key}'][valid]
            metrics[f'rmse_{var}'] = np.sqrt(np.mean((p - t) ** 2))
            metrics[f'mae_{var}'] = np.mean(np.abs(p - t))
            metrics[f'r2_{var}'] = 1 - np.sum((t - p) ** 2) / np.sum((t - np.mean(t)) ** 2)
        else:
            metrics[f'rmse_{var}'] = np.nan

    logger.info(f"测试结果 {experiment_id}: {metrics}")

    df_res = pd.DataFrame({'id': res['id'], 'time': res['time']})
    for i in range(DataEfficiencyConfig.spatial_points):
        df_res[f'pred_hs_{i}'] = res['pred_h'][:, i]
        df_res[f'pred_tm_{i}'] = res['pred_tm'][:, i]
    df_res.to_excel(paths['train_test_result_dir'] / f"results_{experiment_id}.xlsx", index=False)

    return metrics


def main():
    logger.info("启动数据效率对比实验 (Data Efficiency Experiment) [Hs + Tm]")
    all_metrics = []

    for exp_id, config in DataEfficiencyConfig.experiments.items():
        try:
            model, paths, scaler = train_scaling_experiment(exp_id, config)
            if model:
                metrics = test_experiment(exp_id, model, paths, scaler)
                metrics['experiment'] = config['name']
                metrics['ratio'] = config['ratio']
                metrics['use_virtual'] = config['use_virtual']
                all_metrics.append(metrics)
        except Exception as e:
            logger.error(f"实验 {exp_id} 失败: {str(e)}")
            import traceback
            traceback.print_exc()

    if all_metrics:
        df_summary = pd.DataFrame(all_metrics)
        df_summary.to_excel("data_efficiency_summary.xlsx", index=False)
        logger.info("实验完成，汇总报告: data_efficiency_summary.xlsx")
        print(df_summary)


if __name__ == "__main__":
    main()