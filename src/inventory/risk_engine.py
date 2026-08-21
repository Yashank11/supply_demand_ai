import numpy as np
import pandas as pd
from typing import Dict, Any, List
from scipy.stats import norm

class InventoryRiskEngine:
    """
    Computes dynamic safety stock, reorder points, days of inventory,
    stockout probabilities, and excess/expiry risks.
    """
    def __init__(self, service_level: float = 0.95):
        self.service_level = service_level
        self.z_score = norm.ppf(service_level) # ~1.645 for 95%

    def compute_sku_inventory_health(
        self,
        current_on_hand: int,
        in_transit: int,
        daily_demand_mean: float,
        daily_demand_std: float,
        lead_time_mean: float,
        lead_time_std: float,
        unit_cost: float,
        selling_price: float,
        shelf_life_days: int
    ) -> Dict[str, Any]:
        mu_d = max(1.0, float(daily_demand_mean))
        sigma_d = max(0.5, float(daily_demand_std))
        L = max(1.0, float(lead_time_mean))
        sigma_L = max(0.1, float(lead_time_std))
        
        # Dual variance Safety Stock formula
        # SS = Z * sqrt(L * sigma_d^2 + mu_d^2 * sigma_L^2)
        variance_term = (L * (sigma_d ** 2)) + ((mu_d ** 2) * (sigma_L ** 2))
        safety_stock = int(np.ceil(self.z_score * np.sqrt(variance_term)))
        
        # Lead time demand & Reorder Point
        lead_time_demand = int(np.ceil(mu_d * L))
        reorder_point = lead_time_demand + safety_stock
        
        # Days of Inventory (DOI)
        effective_inventory = current_on_hand + in_transit
        days_on_hand = round(current_on_hand / mu_d, 1)
        total_doi = round(effective_inventory / mu_d, 1)
        
        # Probabilistic Stockout Risk calculation
        # Risk = P(Demand during lead time > Current Available Inventory)
        lead_time_demand_std = np.sqrt(variance_term)
        if lead_time_demand_std > 0:
            z_curr = (effective_inventory - lead_time_demand) / lead_time_demand_std
            # P(Demand > effective_inventory) = 1 - Phi(z_curr)
            stockout_prob = float(np.clip(1.0 - norm.cdf(z_curr), 0.01, 0.99))
        else:
            stockout_prob = 0.5 if effective_inventory < lead_time_demand else 0.05
            
        stockout_prob_pct = round(stockout_prob * 100.0, 1)
        
        # Days until stockout
        days_until_stockout = max(0.0, round(current_on_hand / mu_d, 1))
        
        # Classification & Status
        if current_on_hand == 0:
            status = "CRITICAL_STOCKOUT"
            status_label = "Stocked Out"
            risk_color = "#FF4B4B" # Red
        elif days_until_stockout <= L or stockout_prob_pct >= 65.0:
            status = "HIGH_RISK"
            status_label = "High Stockout Risk"
            risk_color = "#FF8C00" # Orange-Red
        elif effective_inventory <= reorder_point:
            status = "REORDER_TRIGGERED"
            status_label = "Reorder Needed"
            risk_color = "#FFD700" # Yellow
        elif total_doi > 45.0:
            status = "OVERSTOCK"
            status_label = "Excess Inventory"
            risk_color = "#1E90FF" # Blue
        else:
            status = "HEALTHY"
            status_label = "Optimal"
            risk_color = "#00CC96" # Green
            
        # Expiry risk for perishable products
        expiry_risk = False
        if shelf_life_days < 90 and total_doi > (shelf_life_days * 0.7):
            expiry_risk = True
            
        # Financial capital locked & potential revenue loss
        capital_locked = round(effective_inventory * unit_cost, 2)
        revenue_at_risk = round(max(0, lead_time_demand - effective_inventory) * selling_price, 2)
        
        return {
            "current_on_hand": current_on_hand,
            "in_transit": in_transit,
            "effective_inventory": effective_inventory,
            "safety_stock": safety_stock,
            "lead_time_demand": lead_time_demand,
            "reorder_point": reorder_point,
            "days_until_stockout": days_until_stockout,
            "days_of_inventory": total_doi,
            "stockout_probability_pct": stockout_prob_pct,
            "status": status,
            "status_label": status_label,
            "risk_color": risk_color,
            "expiry_risk": expiry_risk,
            "capital_locked_inr": capital_locked,
            "revenue_at_risk_inr": revenue_at_risk
        }

    def analyze_network_inventory(self, master_data: Dict[str, Any]) -> pd.DataFrame:
        daily_df = master_data["daily_demand"]
        suppliers_dict = master_data["suppliers"].set_index("supplier_id").to_dict(orient="index")
        
        # Get latest day records for each SKU x Warehouse
        latest_date = daily_df["date"].max()
        latest_df = daily_df[daily_df["date"] == latest_date].copy()
        
        # Historical 30-day stats for demand mean and std
        past_30_date = latest_date - pd.Timedelta(days=30)
        recent_history = daily_df[daily_df["date"] >= past_30_date]
        
        stats_df = recent_history.groupby(["sku_id", "warehouse_id"])["demand"].agg(
            demand_mean="mean",
            demand_std="std"
        ).reset_index()
        
        merged = latest_df.merge(stats_df, on=["sku_id", "warehouse_id"], how="left")
        
        results = []
        for _, row in merged.iterrows():
            sup_id = row.get("primary_supplier", "SUP_01")
            sup = suppliers_dict.get(sup_id, {"avg_lead_time_days": 7.0, "lead_time_std_days": 1.5})
            
            health = self.compute_sku_inventory_health(
                current_on_hand=int(row.get("closing_inventory", 0)),
                in_transit=int(row.get("in_transit", 0)),
                daily_demand_mean=float(row.get("demand_mean", 50.0)),
                daily_demand_std=float(row.get("demand_std", 15.0)),
                lead_time_mean=float(sup.get("avg_lead_time_days", 7.0)),
                lead_time_std=float(sup.get("lead_time_std_days", 1.5)),
                unit_cost=float(row.get("unit_cost", 100.0)),
                selling_price=float(row.get("selling_price", 150.0)),
                shelf_life_days=int(row.get("shelf_life_days", 365))
            )
            
            res_row = {
                "sku_id": row["sku_id"],
                "product_name": row["product_name"],
                "category": row["category"],
                "warehouse_id": row["warehouse_id"],
                "primary_supplier": sup_id,
                "unit_cost": row["unit_cost"],
                "selling_price": row["selling_price"],
                **health
            }
            results.append(res_row)
            
        return pd.DataFrame(results)
