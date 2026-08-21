import os
import random
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from src.config import CACHE_DIR

class SupplyChainDataGenerator:
    """
    Generates rich, enterprise-grade multi-echelon supply chain data:
    50 SKUs across 10 Warehouses, 15 Suppliers, 2 years of daily operational history.
    """
    def __init__(self, n_products=40, n_warehouses=10, n_suppliers=15, n_days=365*2, seed=42):
        self.n_products = n_products
        self.n_warehouses = n_warehouses
        self.n_suppliers = n_suppliers
        self.n_days = n_days
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        self.start_date = datetime(2024, 1, 1)
        self.end_date = self.start_date + timedelta(days=self.n_days - 1)

    def generate_warehouses(self) -> pd.DataFrame:
        warehouse_data = [
            ("WH_MUM_01", "Mumbai Central Hub", "Maharashtra", 19.0760, 72.8777, 150000, "Tier-1 Mega"),
            ("WH_DEL_02", "Delhi NCR Distribution Center", "Delhi", 28.7041, 77.1025, 180000, "Tier-1 Mega"),
            ("WH_BLR_03", "Bengaluru Tech Logistics Park", "Karnataka", 12.9716, 77.5946, 120000, "Tier-1 Mega"),
            ("WH_CHE_04", "Chennai Coastal Terminal", "Tamil Nadu", 13.0827, 80.2707, 110000, "Tier-1 Hub"),
            ("WH_KOL_05", "Kolkata Eastern Gateway", "West Bengal", 22.5726, 88.3639, 95000, "Tier-1 Hub"),
            ("WH_HYD_06", "Hyderabad Central Depot", "Telangana", 17.3850, 78.4867, 100000, "Tier-1 Hub"),
            ("WH_PUN_07", "Pune Industrial Depot", "Maharashtra", 18.5204, 73.8567, 85000, "Tier-2 Hub"),
            ("WH_AHM_08", "Ahmedabad Western Hub", "Gujarat", 23.0225, 72.5714, 90000, "Tier-2 Hub"),
            ("WH_JAI_09", "Jaipur Transit Logistics", "Rajasthan", 26.9124, 75.7873, 70000, "Tier-2 Regional"),
            ("WH_LKO_10", "Lucknow North Central Depot", "Uttar Pradesh", 26.8467, 80.9462, 75000, "Tier-2 Regional"),
            ("WH_KOC_11", "Kochi Maritime Gateway", "Kerala", 9.9312, 76.2673, 60000, "Tier-2 Regional"),
            ("WH_GAU_12", "Guwahati North-East Hub", "Assam", 26.1445, 91.7362, 50000, "Tier-3 Frontier"),
            ("WH_IND_13", "Indore Central Depot", "Madhya Pradesh", 22.7196, 75.8577, 65000, "Tier-2 Regional"),
            ("WH_CHD_14", "Chandigarh Express Hub", "Punjab/Haryana", 30.7333, 76.7794, 60000, "Tier-2 Regional"),
            ("WH_PAT_15", "Patna Eastern Hub", "Bihar", 25.5941, 85.1376, 55000, "Tier-3 Frontier"),
            ("WH_NAG_16", "Nagpur Multi-Modal Hub", "Maharashtra", 21.1458, 79.0882, 80000, "Tier-2 Regional"),
            ("WH_SUR_17", "Surat Express Depot", "Gujarat", 21.1702, 72.8311, 65000, "Tier-2 Regional"),
            ("WH_VIZ_18", "Visakhapatnam Coastal Depot", "Andhra Pradesh", 17.6868, 83.2185, 60000, "Tier-2 Regional"),
            ("WH_COI_19", "Coimbatore South Hub", "Tamil Nadu", 11.0168, 76.9558, 55000, "Tier-3 Frontier"),
            ("WH_BHU_20", "Bhubaneswar East Depot", "Odisha", 20.2961, 85.8245, 50000, "Tier-3 Frontier"),
        ]
        selected = warehouse_data[:self.n_warehouses]
        return pd.DataFrame(selected, columns=["warehouse_id", "warehouse_name", "state", "latitude", "longitude", "capacity_units", "tier"])

    def generate_suppliers(self) -> pd.DataFrame:
        supplier_pool = [
            ("SUP_01", "Bharat Sourcing and Logistics", "Mumbai", 6.0, 1.2, 0.95, 4.8),
            ("SUP_02", "Tata Global Component Supply", "Pune", 7.5, 1.8, 0.91, 4.7),
            ("SUP_03", "Reliance Poly and Materials", "Jamnagar", 9.0, 2.5, 0.82, 4.2),
            ("SUP_04", "Sun Pharma Distribution Hub", "Vadodara", 5.0, 0.8, 0.97, 4.9),
            ("SUP_05", "Godrej Agro and FMCG Supply", "Delhi", 6.5, 1.5, 0.88, 4.4),
            ("SUP_06", "Foxconn Electronics Assemblers", "Chennai", 12.0, 3.2, 0.74, 3.9),
            ("SUP_07", "Hindustan Consumer Staples", "Bengaluru", 5.5, 1.1, 0.94, 4.6),
            ("SUP_08", "Mahindra Logistics SCM", "Nashik", 8.0, 2.1, 0.86, 4.3),
            ("SUP_09", "ITC Agro-Produce Network", "Kolkata", 7.0, 1.9, 0.89, 4.5),
            ("SUP_10", "Apollo Healthcare Logistics", "Hyderabad", 4.5, 0.7, 0.98, 4.9),
            ("SUP_11", "Adani Port Logistics Freight", "Mundra", 10.5, 3.5, 0.76, 4.0),
            ("SUP_12", "Havells Electrical Solutions", "Noida", 8.5, 2.0, 0.87, 4.4),
            ("SUP_13", "Amul Dairy Supply Chain", "Anand", 3.5, 0.6, 0.96, 4.8),
            ("SUP_14", "Dabur Naturals and Wellness", "Ghaziabad", 6.0, 1.4, 0.92, 4.6),
            ("SUP_15", "Bajaj Auto Parts SCM", "Aurangabad", 9.5, 2.8, 0.79, 4.1),
        ]
        selected = supplier_pool[:self.n_suppliers]
        return pd.DataFrame(selected, columns=["supplier_id", "supplier_name", "origin_city", "avg_lead_time_days", "lead_time_std_days", "reliability_otif", "quality_score"])

    def generate_products(self, suppliers_df: pd.DataFrame) -> pd.DataFrame:
        categories = {
            "FMCG": [("Beverages", 40, 120, 180), ("Packaged Food", 60, 220, 120), ("Personal Care", 150, 450, 365), ("Home Detergents", 180, 520, 540)],
            "Electronics": [("Smartphones", 8000, 28000, 730), ("Smart Wearables", 1800, 5500, 540), ("Audio Accessories", 900, 3200, 540), ("Home Appliances", 4500, 16000, 730)],
            "Groceries": [("Edible Oil and Ghee", 140, 320, 180), ("Basmati Rice and Grains", 80, 210, 240), ("Dairy Essentials", 35, 75, 20), ("Organic Spices", 120, 380, 300)],
            "Pharmaceuticals": [("Essential Antibiotics", 110, 350, 540), ("Cold and Flu Relief", 45, 160, 365), ("Cardio and Diabetes Care", 220, 680, 730), ("Daily Multivitamins", 95, 310, 540)],
            "Industrial": [("Automotive Lubricants", 320, 950, 730), ("Fasteners and Bearings", 75, 240, 730), ("Safety Equipment", 250, 780, 730), ("Packaging Boxes", 25, 70, 365)]
        }
        
        products = []
        cat_keys = list(categories.keys())
        supplier_ids = suppliers_df["supplier_id"].tolist()
        
        for i in range(1, self.n_products + 1):
            sku_id = f"SKU_{i:04d}"
            cat = cat_keys[i % len(cat_keys)]
            subcats = categories[cat]
            subcat, min_cost, max_cost, shelf_life = subcats[i % len(subcats)]
            
            unit_cost = round(random.uniform(min_cost, max_cost), 2)
            margin = random.uniform(0.18, 0.45)
            selling_price = round(unit_cost * (1 + margin), 2)
            holding_cost_annual = round(unit_cost * random.uniform(0.18, 0.26), 2)
            primary_supplier = supplier_ids[i % len(supplier_ids)]
            base_daily_demand = random.randint(25, 200)
            weight_kg = round(random.uniform(0.2, 12.0), 2)
            
            products.append({
                "sku_id": sku_id,
                "product_name": f"{cat} {subcat} Model-{i:03d}",
                "category": cat,
                "sub_category": subcat,
                "unit_cost": unit_cost,
                "selling_price": selling_price,
                "holding_cost_annual": holding_cost_annual,
                "shelf_life_days": shelf_life,
                "weight_kg": weight_kg,
                "primary_supplier": primary_supplier,
                "base_daily_demand": base_daily_demand
            })
            
        return pd.DataFrame(products)

    def generate_calendar_and_weather(self, warehouses_df: pd.DataFrame) -> pd.DataFrame:
        dates = pd.date_range(self.start_date, self.end_date, freq="D")
        records = []
        
        festivals = {
            "2024-03-25": ("Holi", 1.45),
            "2024-04-11": ("Eid-ul-Fitr", 1.40),
            "2024-08-15": ("Independence Day", 1.25),
            "2024-08-19": ("Raksha Bandhan", 1.35),
            "2024-10-12": ("Dussehra", 1.50),
            "2024-11-01": ("Diwali Mega Sale", 2.30),
            "2024-11-02": ("Diwali Day 2", 2.10),
            "2024-12-25": ("Christmas", 1.35),
            "2025-01-01": ("New Year Sale", 1.45),
            "2025-03-14": ("Holi", 1.45),
            "2025-03-31": ("Eid-ul-Fitr", 1.40),
            "2025-08-15": ("Independence Day", 1.25),
            "2025-10-02": ("Gandhi Jayanti", 1.30),
            "2025-10-20": ("Diwali Mega Sale", 2.35),
            "2025-10-21": ("Diwali Day 2", 2.15),
            "2025-12-25": ("Christmas", 1.35),
        }
        
        for dt in dates:
            dt_str = dt.strftime("%Y-%m-%d")
            is_weekend = 1 if dt.weekday() >= 5 else 0
            day_of_week = dt.strftime("%A")
            month = dt.month
            
            is_monsoon = 1 if month in [6, 7, 8, 9] else 0
            is_summer = 1 if month in [4, 5, 6] else 0
            
            fest_name, fest_mult = festivals.get(dt_str, ("None", 1.0))
            is_holiday = 1 if fest_name != "None" or is_weekend == 1 else 0
            
            for _, wh in warehouses_df.iterrows():
                if is_monsoon:
                    rainfall_mm = max(0.0, float(np.random.normal(25.0, 18.0)))
                    temp_c = float(np.random.normal(28.0, 3.0))
                    extreme_weather = 1 if rainfall_mm > 55.0 else 0
                elif is_summer:
                    rainfall_mm = max(0.0, float(np.random.normal(1.5, 4.0)))
                    temp_c = float(np.random.normal(38.0, 4.0))
                    extreme_weather = 1 if temp_c > 44.0 else 0
                else:
                    rainfall_mm = max(0.0, float(np.random.normal(0.5, 2.0)))
                    temp_c = float(np.random.normal(21.0, 5.0))
                    extreme_weather = 0
                    
                records.append({
                    "date": dt,
                    "date_str": dt_str,
                    "warehouse_id": wh["warehouse_id"],
                    "day_of_week": day_of_week,
                    "month": month,
                    "year": dt.year,
                    "is_weekend": is_weekend,
                    "festival_name": fest_name,
                    "festival_multiplier": fest_mult,
                    "is_holiday": is_holiday,
                    "temperature_c": round(temp_c, 1),
                    "rainfall_mm": round(rainfall_mm, 1),
                    "extreme_weather_flag": extreme_weather
                })
                
        return pd.DataFrame(records)

    def generate_full_dataset(self, force_refresh=False):
        cache_path = CACHE_DIR / "scout_master_dataset.pkl"
        if cache_path.exists() and not force_refresh:
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
                
        print("Generating SCOUT master supply chain dataset...")
        warehouses = self.generate_warehouses()
        suppliers = self.generate_suppliers()
        products = self.generate_products(suppliers)
        calendar_weather = self.generate_calendar_and_weather(warehouses)
        
        daily_records = []
        cw_lookup = calendar_weather.set_index(["date_str", "warehouse_id"]).to_dict(orient="index")
        
        active_skus = products.head(self.n_products)
        dates = pd.date_range(self.start_date, self.end_date, freq="D")
        
        state_dict = {}
        for _, prod in active_skus.iterrows():
            for _, wh in warehouses.iterrows():
                key = (prod["sku_id"], wh["warehouse_id"])
                base_dem = prod["base_daily_demand"] * float(np.random.uniform(0.6, 1.4))
                init_inv = int(base_dem * random.uniform(12, 28))
                state_dict[key] = {
                    "inv": init_inv,
                    "in_transit": 0,
                    "base_dem": base_dem,
                    "pipeline_pos": []
                }

        po_records = []
        po_counter = 1000
        
        prod_supplier_map = products.set_index("sku_id")["primary_supplier"].to_dict()
        supplier_dict = suppliers.set_index("supplier_id").to_dict(orient="index")
        
        for dt_idx, dt in enumerate(dates):
            dt_str = dt.strftime("%Y-%m-%d")
            
            for (sku_id, wh_id), state in state_dict.items():
                cw = cw_lookup.get((dt_str, wh_id), {})
                fest_mult = cw.get("festival_multiplier", 1.0)
                is_weekend = cw.get("is_weekend", 0)
                temp = cw.get("temperature_c", 25.0)
                rain = cw.get("rainfall_mm", 0.0)
                extreme_w = cw.get("extreme_weather_flag", 0)
                
                arrived_qty = 0
                remaining_pipeline = []
                for arr_dt, q in state["pipeline_pos"]:
                    if dt >= arr_dt:
                        arrived_qty += q
                    else:
                        remaining_pipeline.append((arr_dt, q))
                state["pipeline_pos"] = remaining_pipeline
                state["in_transit"] = sum([q for _, q in remaining_pipeline])
                
                opening_inv = state["inv"]
                state["inv"] += arrived_qty
                
                has_promo = 1 if (dt_idx % 14 in [5, 6] or fest_mult > 1.2) and random.random() < 0.35 else 0
                promo_discount = 0.15 if has_promo else 0.0
                
                weather_boost = 1.0
                if "Beverages" in sku_id or "Dairy" in sku_id:
                    if temp > 35: weather_boost = 1.30
                elif "Cold and Flu" in sku_id:
                    if rain > 30 or temp < 15: weather_boost = 1.45
                
                day_factor = 1.20 if is_weekend else 0.95
                noise = float(np.random.normal(1.0, 0.12))
                
                daily_demand = max(0, int(state["base_dem"] * fest_mult * day_factor * (1 + promo_discount * 1.5) * weather_boost * noise))
                
                available = state["inv"]
                sales = min(available, daily_demand)
                stockout_units = max(0, daily_demand - available)
                closing_inv = available - sales
                state["inv"] = closing_inv
                
                supplier_id = prod_supplier_map[sku_id]
                sup_info = supplier_dict[supplier_id]
                lead_time = sup_info["avg_lead_time_days"]
                safety_stock = int(1.65 * (state["base_dem"] * 0.3) * np.sqrt(lead_time))
                reorder_point = int(state["base_dem"] * lead_time + safety_stock)
                
                if (closing_inv + state["in_transit"]) <= reorder_point and len(state["pipeline_pos"]) < 2:
                    order_qty = int(state["base_dem"] * lead_time * 2.5)
                    actual_lead = max(2, int(np.random.normal(lead_time, sup_info["lead_time_std_days"])))
                    if extreme_w: actual_lead += random.randint(1, 4)
                    if random.random() > sup_info["reliability_otif"]:
                        actual_lead += random.randint(2, 6)
                        
                    exp_arrival = dt + timedelta(days=int(lead_time))
                    act_arrival = dt + timedelta(days=actual_lead)
                    
                    state["pipeline_pos"].append((act_arrival, order_qty))
                    state["in_transit"] += order_qty
                    
                    po_id = f"PO_{po_counter}"
                    po_counter += 1
                    
                    delay_days = max(0, actual_lead - int(lead_time))
                    status = "Delivered" if act_arrival <= self.end_date else "In-Transit"
                    if delay_days > 0 and status == "In-Transit":
                        status = "Delayed"
                        
                    po_records.append({
                        "po_id": po_id,
                        "sku_id": sku_id,
                        "warehouse_id": wh_id,
                        "supplier_id": supplier_id,
                        "order_date": dt,
                        "order_qty": order_qty,
                        "expected_arrival": exp_arrival,
                        "actual_arrival": act_arrival,
                        "delay_days": delay_days,
                        "status": status
                    })

                daily_records.append({
                    "date": dt,
                    "date_str": dt_str,
                    "sku_id": sku_id,
                    "warehouse_id": wh_id,
                    "opening_inventory": opening_inv,
                    "incoming_inventory": arrived_qty,
                    "demand": daily_demand,
                    "sales": sales,
                    "stockout_units": stockout_units,
                    "closing_inventory": closing_inv,
                    "in_transit": state["in_transit"],
                    "promotion": has_promo,
                    "discount_pct": promo_discount,
                    "festival_multiplier": fest_mult,
                    "temperature_c": temp,
                    "rainfall_mm": rain,
                    "extreme_weather": extreme_w,
                    "is_holiday": cw.get("is_holiday", 0),
                    "day_of_week": cw.get("day_of_week", ""),
                    "month": cw.get("month", 1)
                })

        daily_df = pd.DataFrame(daily_records)
        pos_df = pd.DataFrame(po_records)
        
        prod_meta = products[["sku_id", "product_name", "category", "sub_category", "unit_cost", "selling_price", "holding_cost_annual", "shelf_life_days", "primary_supplier"]]
        daily_df = daily_df.merge(prod_meta, on="sku_id", how="left")
        daily_df["revenue"] = daily_df["sales"] * daily_df["selling_price"] * (1 - daily_df["discount_pct"])
        daily_df["lost_revenue"] = daily_df["stockout_units"] * daily_df["selling_price"]

        master_data = {
            "warehouses": warehouses,
            "suppliers": suppliers,
            "products": products,
            "daily_demand": daily_df,
            "purchase_orders": pos_df,
            "calendar_weather": calendar_weather
        }
        
        with open(cache_path, "wb") as f:
            pickle.dump(master_data, f)
            
        print("SCOUT dataset generated successfully: " + str(len(daily_df)) + " daily records, " + str(len(pos_df)) + " PO records.")
        return master_data
