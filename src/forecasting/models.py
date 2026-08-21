import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List
from datetime import timedelta

class FeatureEngineer:
    @staticmethod
    def extract_features(df: pd.DataFrame, target_col: str = "demand") -> pd.DataFrame:
        df = df.copy().sort_values("date").reset_index(drop=True)
        
        # Lag features
        for lag in [1, 2, 3, 7, 14, 21, 28]:
            df[f"lag_{lag}"] = df[target_col].shift(lag)
            
        # Rolling statistics
        df["rolling_mean_7"] = df[target_col].shift(1).rolling(7).mean()
        df["rolling_std_7"] = df[target_col].shift(1).rolling(7).std()
        df["rolling_mean_14"] = df[target_col].shift(1).rolling(14).mean()
        df["rolling_mean_30"] = df[target_col].shift(1).rolling(30).mean()
        df["rolling_max_7"] = df[target_col].shift(1).rolling(7).max()
        df["rolling_min_7"] = df[target_col].shift(1).rolling(7).min()
        
        # Date & Calendar features
        df["day_of_week_num"] = df["date"].dt.dayofweek
        df["day_of_month"] = df["date"].dt.day
        df["month_num"] = df["date"].dt.month
        df["is_weekend_num"] = df["day_of_week_num"].apply(lambda x: 1 if x >= 5 else 0)
        
        # Seasonal sine/cosine terms
        df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week_num"] / 7)
        df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week_num"] / 7)
        df["sin_month"] = np.sin(2 * np.pi * df["month_num"] / 12)
        df["cos_month"] = np.cos(2 * np.pi * df["month_num"] / 12)
        
        # Fill NAs from lags with backfill/forward fill
        df = df.bfill().ffill()
        return df

class MovingAverageForecaster:
    def __init__(self, window: int = 7):
        self.window = window
        self.last_values = []
        self.std = 1.0

    def fit(self, y: np.ndarray):
        self.last_values = y[-self.window:].tolist()
        self.std = max(1.0, float(np.std(y[-30:])))

    def predict(self, horizon: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean_val = float(np.mean(self.last_values))
        p50 = np.full(horizon, mean_val)
        uncertainty = np.linspace(1.28 * self.std, 1.96 * self.std * 1.5, horizon)
        p10 = np.maximum(0, p50 - uncertainty)
        p90 = p50 + uncertainty
        return p10, p50, p90

class SeasonalForecaster:
    """
    Decomposed Seasonal forecaster with trend, day-of-week, month, and festival multipliers.
    """
    def __init__(self):
        self.base_level = 0.0
        self.trend = 0.0
        self.dow_effects = {}
        self.month_effects = {}
        self.residual_std = 1.0

    def fit(self, df: pd.DataFrame, target_col: str = "demand"):
        df = df.sort_values("date").reset_index(drop=True)
        y = df[target_col].values
        n = len(y)
        x = np.arange(n)
        
        # Linear trend
        p = np.polyfit(x, y, 1)
        self.trend = p[0]
        self.base_level = p[1]
        
        detrended = y / np.maximum(1.0, (self.base_level + self.trend * x))
        df["detrended"] = detrended
        
        self.dow_effects = df.groupby("day_of_week")["detrended"].mean().to_dict()
        self.month_effects = df.groupby("month")["detrended"].mean().to_dict()
        
        fitted = (self.base_level + self.trend * x) * df["day_of_week"].map(self.dow_effects) * df["month"].map(self.month_effects)
        residuals = y - fitted
        self.residual_std = max(1.0, float(np.std(residuals)))

    def predict(self, future_dates_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        horizon = len(future_dates_df)
        p50 = []
        for idx, row in future_dates_df.iterrows():
            dow = row.get("day_of_week", "Monday")
            month = row.get("month", 1)
            fest_mult = row.get("festival_multiplier", 1.0)
            promo_boost = 1.20 if row.get("promotion", 0) else 1.0
            
            dow_m = self.dow_effects.get(dow, 1.0)
            mon_m = self.month_effects.get(month, 1.0)
            
            val = (self.base_level + self.trend * (100 + idx)) * dow_m * mon_m * fest_mult * promo_boost
            p50.append(max(0.0, float(val)))
            
        p50 = np.array(p50)
        uncertainty = np.linspace(1.28 * self.residual_std, 1.96 * self.residual_std * 1.4, horizon)
        p10 = np.maximum(0, p50 - uncertainty)
        p90 = p50 + uncertainty
        return p10, p50, p90

class XGBoostForecaster:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42
        )
        self.feature_cols = []
        self.residual_std = 1.0
        self.feature_importances_ = {}

    def fit(self, df_features: pd.DataFrame, target_col: str = "demand"):
        ignore_cols = ["date", "date_str", "sku_id", "warehouse_id", "product_name", "category", "sub_category", "primary_supplier", "day_of_week", "festival_name", "status", "po_id", target_col]
        self.feature_cols = [c for c in df_features.columns if c not in ignore_cols and np.issubdtype(df_features[c].dtype, np.number)]
        
        X = df_features[self.feature_cols].values
        y = df_features[target_col].values
        
        self.model.fit(X, y)
        preds = self.model.predict(X)
        self.residual_std = max(1.0, float(np.std(y - preds)))
        
        importances = self.model.feature_importances_
        self.feature_importances_ = {col: round(float(imp), 4) for col, imp in zip(self.feature_cols, importances)}
        self.feature_importances_ = dict(sorted(self.feature_importances_.items(), key=lambda item: item[1], reverse=True))

    def predict(self, df_future_features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = df_future_features[self.feature_cols].values
        p50 = np.maximum(0, self.model.predict(X))
        horizon = len(p50)
        uncertainty = np.linspace(1.28 * self.residual_std, 1.96 * self.residual_std * 1.35, horizon)
        p10 = np.maximum(0, p50 - uncertainty)
        p90 = p50 + uncertainty
        return p10, p50, p90

class LSTMNeuralModel(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 32, num_layers: int = 2, output_dim: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.15)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class LSTMForecaster:
    def __init__(self, seq_len: int = 28, epochs: int = 40):
        self.seq_len = seq_len
        self.epochs = epochs
        self.model = LSTMNeuralModel()
        self.mean = 0.0
        self.std = 1.0
        self.residual_std = 1.0

    def fit(self, y_series: np.ndarray):
        self.mean = float(np.mean(y_series))
        self.std = max(1e-4, float(np.std(y_series)))
        norm_y = (y_series - self.mean) / self.std
        
        X_list, y_list = [], []
        for i in range(len(norm_y) - self.seq_len):
            X_list.append(norm_y[i:i+self.seq_len])
            y_list.append(norm_y[i+self.seq_len])
            
        if len(X_list) < 10:
            return
            
        X = torch.tensor(np.array(X_list), dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(-1)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            preds = self.model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            
        self.model.eval()
        with torch.no_grad():
            fitted = self.model(X).squeeze().numpy() * self.std + self.mean
            actuals = y.squeeze().numpy() * self.std + self.mean
            self.residual_std = max(1.0, float(np.std(actuals - fitted)))

    def predict(self, recent_history: np.ndarray, horizon: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        curr_seq = list((recent_history[-self.seq_len:] - self.mean) / self.std)
        preds_norm = []
        
        with torch.no_grad():
            for _ in range(horizon):
                inp = torch.tensor(np.array(curr_seq[-self.seq_len:]), dtype=torch.float32).view(1, self.seq_len, 1)
                next_val = self.model(inp).item()
                preds_norm.append(next_val)
                curr_seq.append(next_val)
                
        p50 = np.maximum(0, np.array(preds_norm) * self.std + self.mean)
        uncertainty = np.linspace(1.28 * self.residual_std, 1.96 * self.residual_std * 1.4, horizon)
        p10 = np.maximum(0, p50 - uncertainty)
        p90 = p50 + uncertainty
        return p10, p50, p90
