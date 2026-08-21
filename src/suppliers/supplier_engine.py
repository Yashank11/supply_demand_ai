import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

class SupplierIntelligenceEngine:
    """
    Evaluates supplier performance, delay risks, bottleneck points,
    and downstream supply chain propagation.
    """
    def __init__(self):
        pass

    def evaluate_suppliers(self, master_data: Dict[str, Any]) -> pd.DataFrame:
        suppliers_df = master_data["suppliers"].copy()
        pos_df = master_data["purchase_orders"].copy()
        
        # Calculate historical stats from purchase orders
        po_stats = pos_df.groupby("supplier_id").agg(
            total_orders=("po_id", "count"),
            delayed_orders=("delay_days", lambda x: (x > 0).sum()),
            avg_delay_days=("delay_days", "mean"),
            max_delay_days=("delay_days", "max"),
            total_quantity_supplied=("order_qty", "sum")
        ).reset_index()
        
        merged = suppliers_df.merge(po_stats, on="supplier_id", how="left").fillna(0)
        
        results = []
        for _, row in merged.iterrows():
            tot_orders = max(1, row["total_orders"])
            delay_freq = row["delayed_orders"] / tot_orders
            avg_delay = float(row["avg_delay_days"])
            lead_std = float(row["lead_time_std_days"])
            base_lead = float(row["avg_lead_time_days"])
            otif = float(row["reliability_otif"])
            quality = float(row["quality_score"])
            
            # Composite Supplier Risk Score (0 = Perfect/Low Risk, 100 = Severe Risk)
            # Weights: OTIF (35%), Delay Frequency (25%), Delay Magnitude (20%), Volatility (10%), Quality (10%)
            otif_penalty = (1.0 - otif) * 100.0 * 0.35
            delay_freq_penalty = delay_freq * 100.0 * 0.25
            delay_mag_penalty = min(100.0, (avg_delay / max(1.0, base_lead)) * 100.0) * 0.20
            volatility_penalty = min(100.0, (lead_std / max(1.0, base_lead)) * 100.0) * 0.10
            quality_penalty = max(0.0, (5.0 - quality) / 5.0 * 100.0) * 0.10
            
            risk_score = round(np.clip(otif_penalty + delay_freq_penalty + delay_mag_penalty + volatility_penalty + quality_penalty, 5.0, 95.0), 1)
            
            # Risk Level Badge
            if risk_score >= 60.0:
                risk_tier = "HIGH_RISK"
                risk_tier_label = "High Risk"
                badge_color = "#FF4B4B"
            elif risk_score >= 35.0:
                risk_tier = "MODERATE_RISK"
                risk_tier_label = "Moderate"
                badge_color = "#FF8C00"
            else:
                risk_tier = "LOW_RISK"
                risk_tier_label = "Reliable"
                badge_color = "#00CC96"
                
            # Estimated Delay Probability for upcoming orders
            delay_prob_pct = round(np.clip((delay_freq * 0.6 + (1 - otif) * 0.4) * 100.0, 5.0, 90.0), 1)
            
            results.append({
                "supplier_id": row["supplier_id"],
                "supplier_name": row["supplier_name"],
                "origin_city": row["origin_city"],
                "avg_lead_time_days": round(base_lead, 1),
                "lead_time_std_days": round(lead_std, 1),
                "otif_reliability_pct": round(otif * 100.0, 1),
                "quality_score": round(quality, 2),
                "total_orders": int(row["total_orders"]),
                "historical_delay_rate_pct": round(delay_freq * 100.0, 1),
                "avg_delay_days": round(avg_delay, 1),
                "max_delay_days": int(row["max_delay_days"]),
                "total_quantity_supplied": int(row["total_quantity_supplied"]),
                "supplier_risk_score": risk_score,
                "risk_tier": risk_tier,
                "risk_tier_label": risk_tier_label,
                "badge_color": badge_color,
                "delay_probability_pct": delay_prob_pct
            })
            
        return pd.DataFrame(results).sort_values("supplier_risk_score", ascending=False).reset_index(drop=True)

    def propagate_supplier_disruption(
        self,
        supplier_id: str,
        simulated_delay_days: int,
        master_data: Dict[str, Any],
        inventory_health_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Propagates a hypothetical supplier delay across downstream warehouses, SKUs, and lost revenue.
        """
        products_df = master_data["products"]
        warehouses_df = master_data["warehouses"]
        
        # Identify SKUs supplied by this supplier
        supplied_skus = products_df[products_df["primary_supplier"] == supplier_id]["sku_id"].tolist()
        
        # Filter inventory health for these SKUs
        affected_inv = inventory_health_df[inventory_health_df["sku_id"].isin(supplied_skus)].copy()
        
        affected_warehouses = affected_inv["warehouse_id"].unique().tolist()
        
        # Calculate impact of extra delay on stockouts
        stockout_count = 0
        total_lost_revenue = 0.0
        affected_items = []
        
        for _, row in affected_inv.iterrows():
            current_days = row["days_until_stockout"]
            normal_lead = row["lead_time_demand"] / max(1.0, row["safety_stock"]) # approx
            
            # If current days of inventory is less than normal lead + simulated delay, stockout occurs
            if current_days < simulated_delay_days:
                stockout_days = simulated_delay_days - current_days
                daily_dem = row["lead_time_demand"] / 7.0 # approx
                lost_units = int(np.ceil(stockout_days * daily_dem))
                lost_rev = round(lost_units * row["selling_price"], 2)
                
                stockout_count += 1
                total_lost_revenue += lost_rev
                
                affected_items.append({
                    "sku_id": row["sku_id"],
                    "product_name": row["product_name"],
                    "category": row["category"],
                    "warehouse_id": row["warehouse_id"],
                    "current_on_hand": row["current_on_hand"],
                    "days_until_stockout": row["days_until_stockout"],
                    "estimated_stockout_units": lost_units,
                    "estimated_lost_revenue_inr": lost_rev
                })
                
        return {
            "supplier_id": supplier_id,
            "simulated_delay_days": simulated_delay_days,
            "total_supplied_skus_count": len(supplied_skus),
            "affected_warehouses_count": len(affected_warehouses),
            "critical_stockout_skus_count": stockout_count,
            "total_lost_revenue_inr": round(total_lost_revenue, 2),
            "affected_items": affected_items
        }

    def build_network_graph_data(self, master_data: Dict[str, Any], top_n_skus: int = 15) -> Dict[str, Any]:
        """
        Creates node and edge payloads for network graph visualization.
        """
        suppliers = master_data["suppliers"]
        warehouses = master_data["warehouses"]
        products = master_data["products"].head(top_n_skus)
        
        nodes = []
        edges = []
        
        # Supplier nodes
        for _, s in suppliers.iterrows():
            nodes.append({
                "id": s["supplier_id"],
                "label": s["supplier_name"][:18] + "..",
                "type": "Supplier",
                "color": "#6366F1", # Indigo
                "size": 25
            })
            
        # Warehouse nodes
        for _, w in warehouses.iterrows():
            nodes.append({
                "id": w["warehouse_id"],
                "label": w["warehouse_name"][:16] + "..",
                "type": "Warehouse",
                "color": "#06B6D4", # Cyan
                "size": 22
            })
            
        # SKU nodes
        for _, p in products.iterrows():
            nodes.append({
                "id": p["sku_id"],
                "label": p["product_name"][:14],
                "type": "SKU",
                "color": "#10B981", # Emerald
                "size": 16
            })
            # Edge: Supplier -> SKU
            edges.append({
                "source": p["primary_supplier"],
                "target": p["sku_id"],
                "label": "Supplies"
            })
            
        # Warehouse linkages (connect SKUs to top 3 warehouses)
        for _, p in products.iterrows():
            for w_id in warehouses["warehouse_id"].head(3):
                edges.append({
                    "source": p["sku_id"],
                    "target": w_id,
                    "label": "Distributed To"
                })
                
        return {"nodes": nodes, "edges": edges}
