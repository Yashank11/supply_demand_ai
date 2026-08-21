import numpy as np
import pandas as pd
from typing import Dict, Any, List

class WhatIfScenarioSimulator:
    """
    Simulates operational supply chain stress tests:
    1. Supplier Delay Shock (e.g. Supplier S3 delayed +N days)
    2. Demand Surge Shock (e.g. +X% festival / promo surge)
    3. Warehouse Disruption / Capacity Shutdown (e.g. Mumbai flooded, -Y% throughput)
    4. Lead Time Volatility Shock
    """
    def __init__(self):
        pass

    def run_multi_scenario_simulation(
        self,
        master_data: Dict[str, Any],
        inventory_health_df: pd.DataFrame,
        supplier_id_shock: str = "ALL",
        supplier_delay_days: int = 5,
        demand_surge_pct: float = 25.0,
        warehouse_disruption_id: str = "NONE",
        warehouse_capacity_loss_pct: float = 40.0
    ) -> Dict[str, Any]:
        """
        Executes comprehensive scenario simulation and quantifies revenue/operational impacts.
        """
        base_inv = inventory_health_df.copy()
        
        sim_results = []
        total_baseline_lost_revenue = float(base_inv["revenue_at_risk_inr"].sum())
        simulated_total_lost_revenue = 0.0
        baseline_critical_stockouts = int((base_inv["status"].isin(["CRITICAL_STOCKOUT", "HIGH_RISK"])).sum())
        simulated_critical_stockouts = 0
        
        for _, row in base_inv.iterrows():
            sku = row["sku_id"]
            wh = row["warehouse_id"]
            sup = row["primary_supplier"]
            curr_on_hand = float(row["current_on_hand"])
            in_transit = float(row["in_transit"])
            base_doi = float(row["days_until_stockout"])
            selling_price = float(row["selling_price"])
            unit_cost = float(row["unit_cost"])
            
            # Apply Demand Surge Shock
            demand_multiplier = 1.0 + (demand_surge_pct / 100.0)
            
            # Apply Supplier Delay Shock
            extra_delay = 0
            if supplier_id_shock == "ALL" or supplier_id_shock == sup:
                extra_delay = supplier_delay_days
                
            # Apply Warehouse Disruption Shock
            wh_capacity_factor = 1.0
            if warehouse_disruption_id == "ALL" or warehouse_disruption_id == wh:
                wh_capacity_factor = max(0.1, 1.0 - (warehouse_capacity_loss_pct / 100.0))
                
            # Effective simulated daily demand
            effective_daily_dem = (row["lead_time_demand"] / 7.0) * demand_multiplier
            
            # Simulated days until stockout
            sim_doi = max(0.0, curr_on_hand / max(1.0, effective_daily_dem))
            
            # Total lead time required before fresh stock arrives
            total_lead_time_needed = (row["lead_time_demand"] / (row["lead_time_demand"] / 7.0)) + extra_delay
            
            # Check if stockout occurs in simulation
            is_stockout = False
            sim_lost_units = 0
            sim_lost_rev = 0.0
            
            if sim_doi < total_lead_time_needed:
                is_stockout = True
                stockout_days = total_lead_time_needed - sim_doi
                sim_lost_units = int(np.ceil(stockout_days * effective_daily_dem))
                sim_lost_rev = round(sim_lost_units * selling_price, 2)
                simulated_critical_stockouts += 1
                
            simulated_total_lost_revenue += sim_lost_rev
            
            sim_results.append({
                "sku_id": sku,
                "product_name": row["product_name"],
                "category": row["category"],
                "warehouse_id": wh,
                "primary_supplier": sup,
                "baseline_doi": base_doi,
                "simulated_doi": round(sim_doi, 1),
                "baseline_stockout_prob_pct": row["stockout_probability_pct"],
                "simulated_stockout_flag": is_stockout,
                "simulated_lost_units": sim_lost_units,
                "simulated_lost_revenue_inr": sim_lost_rev
            })
            
        sim_df = pd.DataFrame(sim_results)
        
        # Delta Metrics
        delta_lost_revenue = max(0.0, simulated_total_lost_revenue - total_baseline_lost_revenue)
        delta_stockouts = max(0, simulated_critical_stockouts - baseline_critical_stockouts)
        
        # Top 10 most severely impacted items
        top_impacted = sim_df[sim_df["simulated_lost_revenue_inr"] > 0].sort_values(
            "simulated_lost_revenue_inr", ascending=False
        ).head(10).to_dict(orient="records")
        
        # Summary by Category
        cat_summary = sim_df.groupby("category").agg(
            total_lost_revenue=("simulated_lost_revenue_inr", "sum"),
            stockout_items=("simulated_stockout_flag", "sum")
        ).reset_index()
        
        # Summary by Warehouse
        wh_summary = sim_df.groupby("warehouse_id").agg(
            total_lost_revenue=("simulated_lost_revenue_inr", "sum"),
            stockout_items=("simulated_stockout_flag", "sum")
        ).reset_index()
        
        return {
            "scenario_params": {
                "supplier_id_shock": supplier_id_shock,
                "supplier_delay_days": supplier_delay_days,
                "demand_surge_pct": demand_surge_pct,
                "warehouse_disruption_id": warehouse_disruption_id,
                "warehouse_capacity_loss_pct": warehouse_capacity_loss_pct
            },
            "baseline_lost_revenue_inr": round(total_baseline_lost_revenue, 2),
            "simulated_total_lost_revenue_inr": round(simulated_total_lost_revenue, 2),
            "delta_lost_revenue_inr": round(delta_lost_revenue, 2),
            "baseline_critical_stockouts": baseline_critical_stockouts,
            "simulated_critical_stockouts": simulated_critical_stockouts,
            "delta_critical_stockouts": delta_stockouts,
            "top_impacted_items": top_impacted,
            "category_impact": cat_summary.to_dict(orient="records"),
            "warehouse_impact": wh_summary.to_dict(orient="records"),
            "full_sim_df": sim_df
        }
