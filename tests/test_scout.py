import pytest
import numpy as np
import pandas as pd
from src.data.generator import SupplyChainDataGenerator
from src.forecasting.models import (
    FeatureEngineer,
    MovingAverageForecaster,
    SeasonalForecaster,
    XGBoostForecaster
)
from src.forecasting.evaluator import ForecastEvaluator
from src.inventory.risk_engine import InventoryRiskEngine
from src.suppliers.supplier_engine import SupplierIntelligenceEngine
from src.decision.recommender import DecisionRecommendationEngine
from src.decision.simulator import WhatIfScenarioSimulator

@pytest.fixture(scope='session')
def master_data():
    gen = SupplyChainDataGenerator(n_products=10, n_warehouses=4, n_suppliers=5, n_days=180)
    return gen.generate_full_dataset(force_refresh=True)

def test_data_generation(master_data):
    assert 'warehouses' in master_data
    assert 'suppliers' in master_data
    assert 'products' in master_data
    assert 'daily_demand' in master_data
    assert 'purchase_orders' in master_data
    
    assert len(master_data['warehouses']) == 4
    assert len(master_data['suppliers']) == 5
    assert len(master_data['products']) == 10
    assert len(master_data['daily_demand']) > 0

def test_forecasting_models(master_data):
    df = master_data['daily_demand']
    sample = df[(df['sku_id'] == 'SKU_0001') & (df['warehouse_id'] == df['warehouse_id'].iloc[0])].copy()
    
    # 1. Moving Average
    ma = MovingAverageForecaster(window=7)
    ma.fit(sample['demand'].values)
    p10, p50, p90 = ma.predict(14)
    assert len(p50) == 14
    assert np.all(p90 >= p50) and np.all(p50 >= p10)
    
    # 2. XGBoost
    feat_df = FeatureEngineer.extract_features(sample, 'demand')
    xgb = XGBoostForecaster()
    xgb.fit(feat_df, 'demand')
    x_p10, x_p50, x_p90 = xgb.predict(feat_df.tail(14))
    assert len(x_p50) == 14
    assert np.all(x_p90 >= x_p50) and np.all(x_p50 >= x_p10)

def test_inventory_risk_engine(master_data):
    engine = InventoryRiskEngine(service_level=0.95)
    health_df = engine.analyze_network_inventory(master_data)
    assert len(health_df) > 0
    assert 'stockout_probability_pct' in health_df.columns
    assert 'safety_stock' in health_df.columns
    assert 'reorder_point' in health_df.columns
    assert 'days_until_stockout' in health_df.columns
    assert np.all(health_df['stockout_probability_pct'] >= 0)
    assert np.all(health_df['stockout_probability_pct'] <= 100)

def test_supplier_engine(master_data):
    sup_engine = SupplierIntelligenceEngine()
    sup_eval = sup_engine.evaluate_suppliers(master_data)
    assert len(sup_eval) == len(master_data['suppliers'])
    assert 'supplier_risk_score' in sup_eval.columns
    assert 'otif_reliability_pct' in sup_eval.columns

def test_decision_and_simulation_engine(master_data):
    inv_engine = InventoryRiskEngine(service_level=0.95)
    health_df = inv_engine.analyze_network_inventory(master_data)
    
    sup_engine = SupplierIntelligenceEngine()
    sup_eval = sup_engine.evaluate_suppliers(master_data)
    
    dec_engine = DecisionRecommendationEngine()
    recs = dec_engine.generate_recommendations(health_df, sup_eval, master_data['warehouses'])
    assert isinstance(recs, list)
    
    sim = WhatIfScenarioSimulator()
    sim_out = sim.run_multi_scenario_simulation(
        master_data,
        health_df,
        supplier_id_shock='ALL',
        supplier_delay_days=4,
        demand_surge_pct=30.0
    )
    assert 'simulated_total_lost_revenue_inr' in sim_out
    assert 'simulated_critical_stockouts' in sim_out
    assert sim_out['simulated_total_lost_revenue_inr'] >= 0
