# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
import pickle
from tqdm import tqdm
import re
import math
import glob
import gc
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class PredictionConfig:
    """SWAN-TC 预测配置"""

    # 模型架构参数 (需与 SWAN-T/train.py 中的 DataEfficiencyConfig 一致)
    d_model = 256
    nhead = 8
    num_layers = 6
    dim_feedforward = 512
    dropout = 0.1
    activation = 'gelu'

    input_channels = {
        'spatial': 1,
        'physical': 7,
        'output': 2
    }

    # 空间网格配置
    coord_cols = ['depth']
    depth_indices = list(range(10, 200, 3))  # 64 points
    spatial_points = len(depth_indices)
    grid_spacing = 50

    # 预测年份与月份
    predict_years = [2021, 2022, 2023]
    test_months = list(range(1, 13))

    VALID_BUOYS = [
        "Bos_waves", "HgI_waves", "ChP_waves", "Mlf_waves", "BkB_waves", "Clv_waves", "Csl_waves", "Dwl_waves",
        "Flk_waves", "GdS_waves", "Hpg_waves", "LoB_waves", "Nbg_waves", "Plv_waves", "Pnz_waves", "Prp_waves",
        "RhF_waves", "Rst_waves", "SBy_waves", "Sca_waves", "StB_waves", "Wtb_waves", "Bdf_waves", "Hrn_waves",
        "Fxs_waves", "PBy_waves", "WBy_waves"
    ]

    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 路径配置 (请根据本地环境修改)
    data_files_2021 = {
        "depth": r"E:\profile\UK_year\swan_true\model_input\swan_depth_2021.csv",
        "wind": r"E:\profile\UK_year\swan_true\model_input\wind_wave_initial_data_2021.csv",
    }
    data_files_2022 = {
        "depth": r"E:\profile\UK_year\swan_true\model_input\swan_depth_2022.csv",
        "wind": r"E:\profile\UK_year\swan_true\model_input\wind_wave_initial_data_2022.csv",
    }
    data_files_2023 = {
        "depth": r"E:\profile\UK_year\swan_true\model_input\swan_depth_2023.csv",
        "wind": r"E:\profile\UK_year\swan_true\model_input\wind_wave_initial_data_2023.csv",
    }

    # SWAN-C 偏差预测结果目录 (运行 SWAN-C/predict.py 后生成)
    bias_file_dir = r"E:\nn_model_v2\bias_invert\bias_profile\results_multi_bias_pred"
    # SWAN-T 模型权重与 Scaler
    model_path = r"E:\nn_model_v2\nn_model\transformer_virtual\best_model_exp_5_100_percent.pth"
    scaler_path = r"E:\nn_model_v2\nn_model\transformer_virtual\results_data_efficiency_exp_5_100_percent\scaler_exp_5_100_percent.pkl"
    result_dir = "./results_swan_tc"


class SpatialCoordManager:
    @classmethod
    def load_coordinates(cls, depth_df: pd.DataFrame, months_filter=None):
        depth_map = {}
        pattern = re.compile(r'^(depth)(\d{3})$')

        if 'id' not in depth_df.columns or 'time' not in depth_df.columns:
            return {'x': [], 'y': [], 'depth_map': {}}

        depth_df['time'] = pd.to_datetime(depth_df['time'], format='%Y%m%d%H', errors='coerce').dt.floor('h')
        depth_df = depth_df.dropna(subset=['time'])

        if months_filter:
            depth_df = depth_df[depth_df['time'].dt.month.isin(months_filter)]

        coord_map = {}
        for col in depth_df.columns:
            if col in ['id', 'time']:
                continue
            match = pattern.match(col)
            if match:
                var_type, idx = match.groups()
                idx_num = int(idx)
                if var_type in PredictionConfig.coord_cols and idx_num in PredictionConfig.depth_indices:
                    coord_map[idx] = {var_type: col}

        for buoy_id in depth_df['id'].unique():
            buoy_df = depth_df[depth_df['id'] == buoy_id]
            depth_map[buoy_id] = {}
            for _, row in buoy_df.iterrows():
                depth_values = []
                for idx_num in PredictionConfig.depth_indices:
                    idx_str = f"{idx_num:03d}"
                    val = row[coord_map[idx_str]['depth']] if idx_str in coord_map else np.nan
                    depth_values.append(val)
                depth_map[buoy_id][row['time']] = np.array(depth_values, dtype=np.float32)

        x = PredictionConfig.grid_spacing * np.array(PredictionConfig.depth_indices, dtype=np.float32)
        y = np.zeros(PredictionConfig.spatial_points, dtype=np.float32)
        return {'x': x, 'y': y, 'depth_map': depth_map}


class SimpleFeatureProcessor:
    @staticmethod
    def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in ['alpc', 'tide']:
            if col not in df.columns:
                df[col] = 0.0
        feature_cols = ['swh', 'wind_speed', 'wind_direction', 'mwd', 'mwp', 'alpc', 'tide']
        for col in feature_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
        return df


class ScenarioScaler:
    def __init__(self):
        self.hs_scaler = StandardScaler()
        self.tm_scaler = StandardScaler()
        self.depth_scaler = StandardScaler()
        self.scenario_scalers = {}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        hs_cols = [f'hs{idx:03d}' for idx in PredictionConfig.depth_indices]
        valid_hs = [c for c in hs_cols if c in df.columns]
        if valid_hs and hasattr(self.hs_scaler, 'mean_'):
            df[valid_hs] = self.hs_scaler.transform(df[valid_hs].fillna(0).astype(np.float32))
        tm_cols = [f'tm{idx:03d}' for idx in PredictionConfig.depth_indices]
        valid_tm = [c for c in tm_cols if c in df.columns]
        if valid_tm and hasattr(self.tm_scaler, 'mean_'):
            df[valid_tm] = self.tm_scaler.transform(df[valid_tm].fillna(0).astype(np.float32))
        depth_cols = [f'depth{idx:03d}' for idx in PredictionConfig.depth_indices]
        valid_depth = [c for c in depth_cols if c in df.columns]
        if valid_depth and hasattr(self.depth_scaler, 'mean_'):
            df[valid_depth] = self.depth_scaler.transform(df[valid_depth].fillna(0).astype(np.float32))
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=PredictionConfig.dropout)
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
        self.spatial_projection = nn.Linear(PredictionConfig.input_channels['spatial'], d_model)
        self.physical_projection = nn.Linear(PredictionConfig.input_channels['physical'], d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(PredictionConfig.dropout)
        self.modal_weights = nn.Parameter(torch.ones(2))

    def forward(self, spatial, physical, spatial_mask, physical_mask):
        spatial = spatial.transpose(1, 2)
        physical = physical.transpose(1, 2)
        spatial_emb = self.spatial_projection(spatial)
        physical_emb = self.physical_projection(physical)
        weights = torch.softmax(self.modal_weights, dim=0)
        combined = (weights[0] * spatial_emb + weights[1] * physical_emb)
        combined = self.dropout(self.layer_norm(combined))
        if spatial_mask.dim() == 3:
            spatial_valid = spatial_mask.squeeze(1)
        else:
            spatial_valid = spatial_mask
        if physical_mask.dim() == 2:
            physical_valid = physical_mask.all(dim=1, keepdim=True).expand(-1, PredictionConfig.spatial_points)
        else:
            physical_valid = physical_mask
        mask = spatial_valid & physical_valid
        return combined, mask


class TransformerWavePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = PredictionConfig.d_model
        self.feature_embedding = FeatureEmbedding(self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=PredictionConfig.spatial_points)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=PredictionConfig.nhead,
                                                   dim_feedforward=PredictionConfig.dim_feedforward,
                                                   dropout=PredictionConfig.dropout,
                                                   activation=PredictionConfig.activation, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=PredictionConfig.num_layers)
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2), nn.GELU(), nn.Dropout(PredictionConfig.dropout),
            nn.Linear(self.d_model // 2, self.d_model // 4), nn.GELU(), nn.Dropout(PredictionConfig.dropout),
            nn.Linear(self.d_model // 4, PredictionConfig.input_channels['output'])
        )

    def forward(self, spatial, physical, spatial_mask, physical_mask):
        emb, mask = self.feature_embedding(spatial, physical, spatial_mask, physical_mask)
        emb = self.pos_encoder(emb.transpose(0, 1)).transpose(0, 1)
        out = self.transformer_encoder(emb, src_key_padding_mask=~mask)
        pred = self.output_projection(out)
        return pred.permute(0, 2, 1)


class PredictionDataset(Dataset):
    def __init__(self, df, coord_system, scaler):
        self.coord_system = coord_system
        self.scaler = scaler
        _df = SimpleFeatureProcessor.add_basic_features(df)

        target_hs_cols = [f'hs{idx:03d}' for idx in PredictionConfig.depth_indices]
        target_tm_cols = [f'tm{idx:03d}' for idx in PredictionConfig.depth_indices]
        all_dummy_cols = target_hs_cols + target_tm_cols

        existing_cols = set(_df.columns)
        missing_cols = [c for c in all_dummy_cols if c not in existing_cols]
        if missing_cols:
            zeros_df = pd.DataFrame(np.zeros((len(_df), len(missing_cols)), dtype=np.float32),
                                    index=_df.index, columns=missing_cols)
            _df = pd.concat([_df, zeros_df], axis=1)

        self.df = self.scaler.transform(_df)
        float_cols = self.df.select_dtypes(include=['float64']).columns
        if len(float_cols) > 0:
            self.df[float_cols] = self.df[float_cols].astype(np.float32)
        del _df
        gc.collect()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        buoy_id = row['id']
        ts = row['time']

        if buoy_id in self.coord_system['depth_map'] and ts in self.coord_system['depth_map'][buoy_id]:
            depth = self.coord_system['depth_map'][buoy_id][ts].astype(np.float32)
        else:
            depth = np.zeros(PredictionConfig.spatial_points, dtype=np.float32)

        spatial_input = depth.reshape(1, -1)
        feats = np.array([
            row['swh'], row['wind_speed'], row['wind_direction'],
            row['mwd'], row['mwp'], row['alpc'], row['tide']
        ], dtype=np.float32)
        physical_input = np.tile(feats, (PredictionConfig.spatial_points, 1)).T
        spatial_mask = ~np.isnan(spatial_input)
        physical_mask = ~np.isnan(feats)

        return (
            torch.FloatTensor(spatial_input),
            torch.FloatTensor(physical_input),
            torch.BoolTensor(spatial_mask),
            torch.BoolTensor(physical_mask),
            buoy_id,
            ts.strftime('%Y%m%d%H')
        )


def get_bias_file_for_year(year):
    search_pattern = os.path.join(PredictionConfig.bias_file_dir, f"results_{year}_*.xlsx")
    files = glob.glob(search_pattern)
    if not files:
        logging.warning(f"未找到 {year} 年的 SWAN-C 偏差文件")
        return None
    return files[0]


def load_bias_data(file_path):
    if not file_path:
        return pd.DataFrame()
    df = pd.read_excel(file_path)
    df = df.rename(columns={'Buoy_ID': 'id', 'Time': 'time'})
    df['time'] = df['time'].astype(str).str.replace(r'\.0$', '', regex=True)
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d%H', errors='coerce')

    if 'Pred_Bias_Hs' not in df.columns:
        df['Pred_Bias_Hs'] = 0.0
    if 'Pred_Bias_Tm' not in df.columns:
        df['Pred_Bias_Tm'] = 0.0

    df = df.dropna(subset=['time'])
    return df[['id', 'time', 'Pred_Bias_Hs', 'Pred_Bias_Tm']]


def main():
    os.makedirs(PredictionConfig.result_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()

    # 加载 SWAN-T 模型
    model = TransformerWavePredictor().to(PredictionConfig.device)
    model.load_state_dict(torch.load(PredictionConfig.model_path, map_location=PredictionConfig.device))
    model.eval()

    with open(PredictionConfig.scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    logger.info("SWAN-T 模型加载成功")

    for year in PredictionConfig.predict_years:
        logger.info(f"\n{'=' * 20} 处理 {year} 年 {'=' * 20}")
        gc.collect()

        # 加载 SWAN-C 偏差预测结果
        bias_file = get_bias_file_for_year(year)
        bias_df = load_bias_data(bias_file)
        if not bias_df.empty:
            bias_df = bias_df[bias_df['id'].isin(PredictionConfig.VALID_BUOYS)]

        files = getattr(PredictionConfig, f"data_files_{year}")
        if not os.path.exists(files['depth']):
            continue

        depth_df = pd.read_csv(files['depth'])
        wind_df = pd.read_csv(files['wind'])

        depth_df = depth_df[depth_df['id'].isin(PredictionConfig.VALID_BUOYS)]
        wind_df = wind_df[wind_df['id'].isin(PredictionConfig.VALID_BUOYS)]

        for d in [depth_df, wind_df]:
            float_cols = d.select_dtypes(include=['float64']).columns
            d[float_cols] = d[float_cols].astype(np.float32)
            d['time'] = pd.to_datetime(d['time'], format='%Y%m%d%H', errors='coerce').dt.floor('h')
            d.drop(d[~d['time'].dt.month.isin(PredictionConfig.test_months)].index, inplace=True)

        input_df = depth_df.merge(wind_df, on=['id', 'time'], how='inner')
        if input_df.empty:
            continue

        temp_depth_df = pd.read_csv(files['depth'])
        temp_depth_df = temp_depth_df[temp_depth_df['id'].isin(PredictionConfig.VALID_BUOYS)]
        coord_sys = SpatialCoordManager.load_coordinates(temp_depth_df, PredictionConfig.test_months)

        del depth_df, wind_df, temp_depth_df
        gc.collect()

        ds = PredictionDataset(input_df, coord_sys, scaler)
        loader = DataLoader(ds, batch_size=PredictionConfig.batch_size, num_workers=0)

        results = {'id': [], 'time': [], 'raw_hs': [], 'raw_tm': [], 'depths': []}

        logger.info(f"SWAN-T 推理 {year}...")
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Predicting {year}"):
                spatial, physical, s_mask, p_mask, ids, times = batch
                spatial = spatial.to(PredictionConfig.device)
                physical = physical.to(PredictionConfig.device)
                s_mask = s_mask.to(PredictionConfig.device)
                p_mask = p_mask.to(PredictionConfig.device)

                output = model(spatial, physical, s_mask, p_mask)
                output = output.cpu().numpy()

                raw_hs = scaler.inverse_transform_targets(output[:, 0, :], 'hs')
                raw_tm = scaler.inverse_transform_targets(output[:, 1, :], 'tm')
                batch_depths = spatial.cpu().numpy()[:, 0, :]

                results['id'].extend(ids)
                results['time'].extend(times)
                results['raw_hs'].append(raw_hs)
                results['raw_tm'].append(raw_tm)
                results['depths'].append(batch_depths)

        del ds, loader
        torch.cuda.empty_cache()
        gc.collect()

        base_df = pd.DataFrame({'id': results['id'], 'time': pd.to_datetime(results['time'], format='%Y%m%d%H')})
        arr_hs = np.concatenate(results['raw_hs'], axis=0)
        arr_tm = np.concatenate(results['raw_tm'], axis=0)
        arr_depth = np.concatenate(results['depths'], axis=0)
        arr_hs = np.maximum(arr_hs, 0)
        arr_tm = np.maximum(arr_tm, 0)

        del results
        gc.collect()

        # 应用 SWAN-C 偏差校正并施加波浪破碎物理约束 (Hs ≤ 0.78 * depth)
        if not bias_df.empty:
            logger.info(f"应用 SWAN-C 偏差校正 ({year})...")
            merged = base_df.merge(bias_df, on=['id', 'time'], how='left')
            bias_val_hs = merged['Pred_Bias_Hs'].fillna(0).values[:, None].astype(np.float32)
            bias_val_tm = merged['Pred_Bias_Tm'].fillna(0).values[:, None].astype(np.float32)
            del merged, bias_df
            gc.collect()

            final_hs = np.maximum(arr_hs - bias_val_hs, 0.0)
            final_tm = np.maximum(arr_tm - bias_val_tm, 0.0)
            breaking_limit = np.maximum(arr_depth * 0.78, 0.0)
            final_hs = np.minimum(final_hs, breaking_limit)
            del bias_val_hs, bias_val_tm, breaking_limit
        else:
            final_hs = arr_hs
            final_tm = arr_tm
            breaking_limit = np.maximum(arr_depth * 0.78, 0.0)
            final_hs = np.minimum(final_hs, breaking_limit)

        # 保存结果
        logger.info(f"保存 {year} 年结果...")
        cols_hs = [f'hs_{i:03d}' for i in PredictionConfig.depth_indices]
        cols_tm = [f'tm_{i:03d}' for i in PredictionConfig.depth_indices]
        base_path_prefix = f"{PredictionConfig.result_dir}/Final_Pred_{year}"

        try:
            pd.concat([base_df, pd.DataFrame(final_hs, columns=cols_hs)], axis=1) \
                .to_csv(f"{base_path_prefix}_Hs_Corrected.csv", index=False)

            pd.concat([base_df, pd.DataFrame(final_tm, columns=cols_tm)], axis=1) \
                .to_csv(f"{base_path_prefix}_Tm_Corrected.csv", index=False)

            pd.concat([base_df, pd.DataFrame(arr_hs, columns=cols_hs)], axis=1) \
                .to_csv(f"{base_path_prefix}_Hs_Raw.csv", index=False)

            pd.concat([base_df, pd.DataFrame(arr_tm, columns=cols_tm)], axis=1) \
                .to_csv(f"{base_path_prefix}_Tm_Raw.csv", index=False)

        except Exception as e:
            logger.error(f"保存失败: {e}")

        logger.info(f"清理 {year} 年内存...")
        del base_df, arr_hs, arr_tm, final_hs, final_tm, arr_depth
        gc.collect()

    logger.info("所有年份处理完成。")


if __name__ == "__main__":
    main()
