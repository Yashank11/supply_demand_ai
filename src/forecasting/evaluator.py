import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.forecasting.models import (
    FeatureEngineer,
    MovingAverageForecaster,
    SeasonalForecaster,
    XGBoostForecaster,
    LSTMForecaster
)

class ForecastEvaluator:
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, unit_cost: float = 100.0, price: float = 150.0) -> Dict[str, float]:
        y_true = np.array(y_true, dtype=float)
        y_pred = np.maximum(0, np.array(y_pred, dtype=float))
        
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        
        sum_true = float(np.sum(y_true))
        wape = float(np.sum(np.abs(y_true - y_pred)) / max(1.0, sum_true)) * 100.0
        
        # sMAPE
        denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        smape = float(np.mean(np.where(denom == 0, 0, np.abs(y_pred - y_true) / denom))) * 100.0
        
        # Business Decision Metric: Estimated Lost Sales + Holding Cost impact
        under_forecasting = np.maximum(0, y_true - y_pred)
        over_forecasting = np.maximum(0, y_pred - y_true)
        lost_margin = float(np.sum(under_forecasting * (price - unit_cost)))
        excess_holding = float(np.sum(over_forecasting * (unit_cost * 0.20 / 365.0 * 14)))
        business_cost = round(lost_margin + excess_holding, 2)
        
        return {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "WAPE_%": round(wape, 2),
            "sMAPE_%": round(smape, 2),
            "Business_Cost_INR": business_cost
        }

    @classmethod
    def benchmark_models_on_series(cls, sku_df: pd.DataFrame, test_days: int = 28) -> pd.DataFrame:
        sku_df = sku_df.sort_values("date").reset_index(drop=True)
        if len(sku_df) <= test_days + 30:
            test_days = max(7, len(sku_df) // 5)
            
        train_df = sku_df.iloc[:-test_days].copy()
        test_df = sku_df.iloc[-test_days:].copy()
        
        y_train = train_df["demand"].values
        y_test = test_df["demand"].values
        unit_cost = float(sku_df["unit_cost"].iloc[0]) if "unit_cost" in sku_df else 100.0
        price = float(sku_df["selling_price"].iloc[0]) if "selling_price" in sku_df else 150.0
        
        results = []
        
        # 1. Moving Average
        t0 = time.time()
        ma_model = MovingAverageForecaster(window=7)
        ma_model.fit(y_train)
        _, ma_preds, _ = ma_model.predict(test_days)
        t_ma = time.time() - t0
        m_ma = cls.calculate_metrics(y_test, ma_preds, unit_cost, price)
        m_ma.update({"Model": "Moving Average (7-Day)", "Training_Time_Sec": round(t_ma, 4)})
        results.append(m_ma)
        
        # 2. Seasonal Forecaster
        t0 = time.time()
        seas_model = SeasonalForecaster()
        seas_model.fit(train_df, "demand")
        _, seas_preds, _ = seas_model.predict(test_df)
        t_seas = time.time() - t0
        m_seas = cls.calculate_metrics(y_test, seas_preds, unit_cost, price)
        m_seas.update({"Model": "Seasonal Decomposition", "Training_Time_Sec": round(t_seas, 4)})
        results.append(m_seas)
        
        # 3. XGBoost Forecaster
        t0 = time.time()
        feat_df = FeatureEngineer.extract_features(sku_df, "demand")
        train_feat = feat_df.iloc[:-test_days].copy()
        test_feat = feat_df.iloc[-test_days:].copy()
        
        xgb_model = XGBoostForecaster()
        xgb_model.fit(train_feat, "demand")
        _, xgb_preds, _ = xgb_model.predict(test_feat)
        t_xgb = time.time() - t0
        m_xgb = cls.calculate_metrics(y_test, xgb_preds, unit_cost, price)
        m_xgb.update({"Model": "XGBoost Regressor", "Training_Time_Sec": round(t_xgb, 4)})
        results.append(m_xgb)
        
        # 4. LSTM Forecaster
        t0 = time.time()
        lstm_model = LSTMForecaster(seq_len=28, epochs=30)
        lstm_model.fit(y_train)
        _, lstm_preds, _ = lstm_model.predict(y_train, test_days)
        t_lstm = time.time() - t0
        m_lstm = cls.calculate_metrics(y_test, lstm_preds, unit_cost, price)
        m_lstm.update({"Model": "PyTorch LSTM Seq2Seq", "Training_Time_Sec": round(t_lstm, 4)})
        results.append(m_lstm)
        
        res_df = pd.DataFrame(results)
        cols = ["Model", "WAPE_%", "MAE", "RMSE", "sMAPE_%", "Business_Cost_INR", "Training_Time_Sec"]
        return res_df[cols].sort_values("WAPE_%").reset_index(drop=True)
