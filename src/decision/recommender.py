import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.config import EXPEDITE_COST_PER_UNIT

class DecisionRecommendationEngine:
    """
    Generates high-priority business recommendations:
    - Purchase Order Reorder Quantities (Economic / Dynamic Min-Max)
    - Expedited Shipment Orders
    - Lateral Inter-Warehouse Inventory Transfers
    - Supplier Diversification
    """
    def __init__(self):
        pass

    def generate_recommendations(
        self,
        inventory_health_df: pd.DataFrame,
        suppliers_df: pd.DataFrame,
        warehouses_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        recommendations = []
        sup_dict = suppliers_df.set_index("supplier_id").to_dict(orient="index")
        wh_dict = warehouses_df.set_index("warehouse_id").to_dict(orient="index")
        
        # Sort inventory items by stockout risk descending
        sorted_inv = inventory_health_df.sort_values(
            by=["stockout_probability_pct", "revenue_at_risk_inr"],
            ascending=[False, False]
        )
        
        for _, row in sorted_inv.iterrows():
            sku = row["sku_id"]
            wh = row["warehouse_id"]
            status = row["status"]
            on_hand = row["current_on_hand"]
            in_transit = row["in_transit"]
            eff_inv = row["effective_inventory"]
            rop = row["reorder_point"]
            lead_dem = row["lead_time_demand"]
            ss = row["safety_stock"]
            stockout_pct = row["stockout_probability_pct"]
            doi = row["days_until_stockout"]
            rev_risk = row["revenue_at_risk_inr"]
            sup_id = row["primary_supplier"]
            sup = sup_dict.get(sup_id, {})
            
            # Action 1: Critical Stockout Immediate Reorder & Expedite
            if status in ["CRITICAL_STOCKOUT", "HIGH_RISK"]:
                target_order = max(100, int((rop * 2.2) - eff_inv))
                expedite_cost = round(target_order * EXPEDITE_COST_PER_UNIT, 2)
                net_savings = round(max(0, rev_risk - expedite_cost), 2)
                
                recommendations.append({
                    "priority": "CRITICAL",
                    "priority_badge": "?? CRITICAL ACTION",
                    "action_type": "EMERGENCY_REORDER_EXPEDITE",
                    "sku_id": sku,
                    "product_name": row["product_name"],
                    "category": row["category"],
                    "warehouse_id": wh,
                    "warehouse_name": wh_dict.get(wh, {}).get("warehouse_name", wh),
                    "supplier_id": sup_id,
                    "supplier_name": sup.get("supplier_name", sup_id),
                    "recommended_order_units": target_order,
                    "estimated_doi_days": doi,
                    "stockout_risk_pct": stockout_pct,
                    "financial_revenue_at_risk_inr": rev_risk,
                    "action_summary": f"Order {target_order:,} units from {sup.get('supplier_name', sup_id)} via Express Freight immediately. Current inventory will exhaust in {doi} days.",
                    "rationale": f"Stockout risk is {stockout_pct}%. Lead time is {sup.get('avg_lead_time_days', 7)} days while inventory lasts only {doi} days. Potential lost revenue: ?{rev_risk:,.0f}."
                })
                
            # Action 2: Reorder Triggered
            elif status == "REORDER_TRIGGERED":
                order_units = max(50, int((rop * 1.8) - eff_inv))
                recommendations.append({
                    "priority": "WARNING",
                    "priority_badge": "?? PLANNED REORDER",
                    "action_type": "STANDARD_REORDER",
                    "sku_id": sku,
                    "product_name": row["product_name"],
                    "category": row["category"],
                    "warehouse_id": wh,
                    "warehouse_name": wh_dict.get(wh, {}).get("warehouse_name", wh),
                    "supplier_id": sup_id,
                    "supplier_name": sup.get("supplier_name", sup_id),
                    "recommended_order_units": order_units,
                    "estimated_doi_days": doi,
                    "stockout_risk_pct": stockout_pct,
                    "financial_revenue_at_risk_inr": rev_risk,
                    "action_summary": f"Place scheduled PO for {order_units:,} units with {sup.get('supplier_name', sup_id)} to replenish safety buffer.",
                    "rationale": f"Effective inventory ({eff_inv}) has breached dynamic Reorder Point ({rop}). Target safety stock: {ss} units."
                })
                
            # Action 3: Overstock / Expiry lateral transfer
            elif status == "OVERSTOCK" and row["days_of_inventory"] > 60:
                excess_units = int(on_hand - rop * 1.5)
                if excess_units > 100:
                    recommendations.append({
                        "priority": "OPTIMIZATION",
                        "priority_badge": "?? INVENTORY TRANSFER",
                        "action_type": "INTER_WAREHOUSE_TRANSFER",
                        "sku_id": sku,
                        "product_name": row["product_name"],
                        "category": row["category"],
                        "warehouse_id": wh,
                        "warehouse_name": wh_dict.get(wh, {}).get("warehouse_name", wh),
                        "supplier_id": sup_id,
                        "supplier_name": sup.get("supplier_name", sup_id),
                        "recommended_order_units": 0,
                        "estimated_doi_days": doi,
                        "stockout_risk_pct": stockout_pct,
                        "financial_revenue_at_risk_inr": 0.0,
                        "action_summary": f"Transfer {excess_units:,} excess units of {sku} from {wh} to deficit sister hubs to reduce holding cost.",
                        "rationale": f"Holding {doi} days of supply (?{row['capital_locked_inr']:,.0f} locked capital). Free up warehouse capacity."
                    })
                    
        return recommendations
