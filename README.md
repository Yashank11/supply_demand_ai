# 🚚 SCOUT — AI Supply Chain Intelligence & Risk Forecasting Platform

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green.svg)](https://xgboost.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-EE4C2C.svg)](https://pytorch.org/)
[![Multi-LLM](https://img.shields.io/badge/AI--Copilot-Groq%20%7C%20Gemini%20%7C%20Mistral-8A2BE2.svg)](#6-ai-supply-chain-copilot)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

**SCOUT** is an enterprise-grade AI decision intelligence layer for supply chains. It combines probabilistic demand forecasting, dynamic safety stock optimization, supplier risk & delay tracking, interactive What-If scenario simulation, and an AI Supply Chain Copilot with multi-provider fallback (**Groq, Google Gemini, Mistral AI, OpenRouter**).

Instead of building merely a demand forecasting model, SCOUT turns time-series prediction into **actionable financial and inventory decisions**:
\text{Demand Forecast} \xrightarrow{\quad} \text{Lead Time Volatility} \xrightarrow{\quad} \text{Stockout Risk \%} \xrightarrow{\quad} \text{Reorder Action} \xrightarrow{\quad} \text{Revenue Protected (₹)}

---

## 🏛️ System Architecture

`
                               ┌───────────────────────────────────────────────┐
                               │       MULTI-SOURCE DATA GENERATOR / DWH       │
                               │  (Sales, Weather, Festivals, POs, Shipments)  │
                               └──────────────────────┬────────────────────────┘
                                                      │
                    ┌─────────────────────────────────┼─────────────────────────────────┐
                    ↓                                 ↓                                 ↓
       ┌──────────────────────────┐     ┌──────────────────────────┐     ┌──────────────────────────┐
       │   DEMAND FORECASTING     │     │  INVENTORY INTELLIGENCE  │     │  SUPPLIER INTELLIGENCE   │
       │  - Moving Average        │     │  - Dynamic Safety Stock  │     │  - Supplier Risk Scoring │
       │  - XGBoost Regressor     │     │  - Reorder Point (ROP)   │     │  - OTIF & Delay Variance │
       │  - Seasonal Prophet Model│     │  - Days of Inventory     │     │  - Delay Probability     │
       │  - PyTorch LSTM Seq2Seq  │     │  - Stockout Risk % (ML)  │     │  - Bottleneck Analytics  │
       │  - P10/P50/P90 Bounds    │     │  - Excess & Expiry Risk  │     │  - Route Weather Impact  │
       └────────────┬─────────────┘     └────────────┬─────────────┘     └────────────┬─────────────┘
                    │                                │                                │
                    └────────────────────────────────┼────────────────────────────────┘
                                                     ↓
                                       ┌──────────────────────────┐
                                       │   DECISION INTELLIGENCE  │
                                       │  - Automated PO Quantity │
                                       │  - Expedite / Transfer   │
                                       │  - Supplier Switching    │
                                       └─────────────┬────────────┘
                                                     │
                                       ┌─────────────┴────────────┐
                                       ↓                          ↓
                        ┌──────────────────────────┐ ┌──────────────────────────┐
                        │   WHAT-IF SIMULATOR      │ │   AI SUPPLY CHAIN COPILOT│
                        │  - Supplier Delay Impact │ │  - Multi-LLM Fallback    │
                        │  - Demand Spike Shock    │ │  - Real-Time RAG Context │
                        │  - Disruption Loss (₹/$) │ │  - Executive Summaries   │
                        └─────────────┬────────────┘ └────────────┬─────────────┘
                                      │                           │
                                      └─────────────┬─────────────┘
                                                    ↓
                               ┌─────────────────────────────────────────┐
                               │   STREAMLIT / MODERN INTERACTIVE UI     │
                               │   (Executive KPIs, Heatmap, Drill-down) │
                               └─────────────────────────────────────────┘
`

---

## 📦 Multi-Source Supply Chain Data Architecture

SCOUT generates a rich, multi-echelon supply chain data warehouse combining real retail sales series with operational logistics variables:

1. **Product Hierarchy (35 SKUs across 5 Categories)**:
   - *FMCG* (Beverages, Packaged Food, Personal Care, Detergents)
   - *Electronics* (Smartphones, Smart Wearables, Audio Accessories, Home Appliances)
   - *Groceries & Perishables* (Edible Oil, Basmati Rice, Dairy Essentials, Spices)
   - *Pharmaceuticals* (Essential Antibiotics, Cold & Flu Relief, Cardio Care, Multivitamins)
   - *Industrial* (Automotive Lubricants, Fasteners, Safety Gear, Packaging)
   - Attributes: SKU ID, unit cost ($\text{₹}$), selling price ($\text{₹}$), margin (\% - 45\%$), shelf life (days), holding cost rate (\% - 26\%$/year), and base daily demand.

2. **Multi-Echelon Network (10 Primary Indian Hubs)**:
   - Mumbai Central Hub, Delhi NCR Distribution Center, Bengaluru Tech Logistics Park, Chennai Coastal Terminal, Kolkata Eastern Gateway, Hyderabad, Pune, Ahmedabad, Jaipur, Lucknow.
   - Tier classifications (Tier-1 Mega, Tier-2 Hub), GPS coordinates, and storage capacity limits.

3. **Supplier Master & Operational Purchase Orders (12 Strategic Suppliers)**:
   - Tata Global Component Supply, Sun Pharma, Reliance Poly & Materials, Foxconn Electronics, Amul Dairy, Adani Port Logistics, Dabur Naturals, Mahindra Logistics, etc.
   - Lead time mean & standard deviation, historical On-Time In-Full (OTIF) rate (\% - 98\%$), quality scores, and active PO shipment tracking.

4. **External Drivers & Indian Festive/Weather Calendar**:
   - Event multipliers: Diwali Mega Sale ($+130\%$), Holi ($+45\%$), Eid ($+40\%$), Dussehra ($+50\%$), Independence Day ($+25\%$), New Year ($+45\%$).
   - Weather signals: Seasonal monsoon rainfall index (causing $+1$ to $+4$ day transit delays, spiking cold & flu medicine demand) and summer heatwaves (spiking beverage demand and spoilage risk).

---

## ⚡ The 5 Intelligence Layers

### Layer 1 — Probabilistic Demand Forecasting
- **Moving Average Baseline**: 7-day and 14-day rolling mean + exponential smoothing.
- **Seasonal Decomposition**: Trend + day-of-week + monthly + festival multipliers.
- **XGBoost Regressor**: Engineered with 18+ features (lag_1 to lag_28, olling_mean_7, olling_std_7, olling_mean_30, price discount %, festival multipliers, weather regressors).
- **PyTorch LSTM Seq2Seq**: Deep neural model mapping a 28-day historical sequence to multi-step future demand.
- **Uncertainty Quantification**: Calculates point forecasts + **P10 (Lower Bound), P50 (Median), and P90 (Upper Bound)** confidence envelopes.

### Layer 2 — Inventory & Stockout Intelligence Engine
- **Dynamic Safety Stock**: Accounts for both demand volatility ($\sigma_d$) and supplier lead-time variance ($\sigma_L$):
  SS = Z \cdot \sqrt{L \cdot \sigma_d^2 + \mu_d^2 \cdot \sigma_L^2}
- **Dynamic Reorder Point (ROP)**:  = \text{Lead Time Demand} + SS$.
- **Days of Inventory (DOI)**: $\text{DOI} = \frac{\text{On Hand} + \text{In Transit}}{\text{Forecast Daily Demand}}$.
- **Stockout Risk Probability (-100\%$)**: Gaussian CDF modeling (\text{Lead Time Demand} > \text{Effective Inventory})$.
- **Days-to-Stockout Countdown**: Real-time days remaining before inventory depletion.

### Layer 3 — Supplier Intelligence & Risk Propagation
- **Supplier Composite Risk Score (-100$)**: Combines OTIF reliability (\%$), delay frequency (\%$), delay magnitude (\%$), lead-time volatility (\%$), and quality (\%$).
- **Bottleneck Propagation Engine**: Traces downstream impact: *Supplier Delay $\rightarrow$ Inbound Shipment Delay $\rightarrow$ Warehouse Stockout $\rightarrow$ Store Stockout $\rightarrow$ Lost Revenue ($\text{₹}$)*.
- **In-Transit PO Tracker**: Active monitoring of delayed purchase orders on water/road.

### Layer 4 — Decision Intelligence & What-If Simulator
- **Actionable Recommendations**: Automated, prioritized operational alerts:
  - 🔴 **CRITICAL ACTION**: Emergency Reorder & Express Freight Expedite.
  - 🟠 **PLANNED REORDER**: Scheduled PO placement at dynamic ROP.
  - 🔵 **INVENTORY TRANSFER**: Inter-warehouse lateral transfer from overstocked hubs to deficit hubs.
- **What-If Scenario Simulator**: Real-time stress testing sliders for:
  - Supplier Delay Shock ($+1$ to $+14$ days)
  - Demand Surge Shock ($-20\%$ to $+80\%$)
  - Warehouse Disruption / Throughput Loss (\%$ to \%$)
  - Instant financial loss quantification ($\text{₹}$ Lakhs) and stockout SKU counts.

### Layer 5 — AI Supply Chain Copilot (Multi-LLM Fallback)
- **Multi-Provider LLM Client**: Resilient automated fallback chain:
  1. **Groq** (llama-3.3-70b-versatile) — Ultra-fast inference
  2. **Google Gemini** (gemini-2.5-flash / gemini-1.5-flash) — Deep analytical reasoning
  3. **Mistral AI** (mistral-large-latest) — High-precision operational summaries
  4. **OpenRouter** (meta-llama/llama-3.3-70b-instruct) — Universal backup
  5. **SCOUT Offline Heuristics Engine** — Failsafe fallback
- **Live RAG Context**: Injects live inventory state, top risk SKUs, supplier scorecards, and simulation results into natural language conversation.

---

## 📊 Streamlit Dashboard Guide (7 Views)

1. **📊 Executive Command Center**: High-level KPIs (*Network Health Score*, *Stockout Risk SKUs*, *Revenue at Risk*, *Capital Locked*), Multi-Warehouse Stockout Heatmap, Inventory Health Donut, and AI Recommended Action Cards.
2. **📈 Demand Forecasting Studio**: SKU & Warehouse pickers, horizon slider (7-30 days), model selector (XGBoost, LSTM, Seasonal, Moving Avg), P10/P50/P90 confidence plot, and XGBoost feature importance breakdown.
3. **📦 Inventory & Stockout Intelligence**: Searchable/filterable SKU grid with Days-to-Stockout countdown, Dynamic Safety Stock, ROP, Health status badges, and holding cost calculators.
4. **🚚 Supplier Intelligence & Network**: Supplier capability radar chart, risk scorecard table, OTIF reliability tracking, and active in-transit PO tracker.
5. **🔮 What-If Scenario Simulator**: Interactive slider controls for supplier delay, demand surge, and warehouse capacity drops with comparative financial loss charts.
6. **🤖 AI Supply Chain Copilot**: Interactive chat assistant with multi-provider engine badge, quick executive prompt shortcuts, and structured 3-step action plans.
7. **⚖️ Model Benchmark & MLOps**: Comparative performance leaderboard (WAPE %, MAE, RMSE, sMAPE, Business Cost $\text{₹}$, Training Latency) and MLOps data drift indicators.

---

## 🏆 Model Performance Benchmark Results

Evaluated on historical hold-out validation dataset (28 days):

| Model | WAPE % | MAE | RMSE | sMAPE % | Business Decision Loss (₹) | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Regressor** | **0.95%** | **1.15** | **1.42** | **0.98%** | **₹17,786** | **1.39s** |
| **PyTorch LSTM Seq2Seq** | **10.15%** | **11.23** | **14.85** | **9.72%** | **₹110,550** | **3.87s** |
| **Seasonal Decomposition** | **15.82%** | **18.34** | **22.10** | **14.65%** | **₹14,679** | **0.02s** |
| **Moving Average (7-Day)** | **16.65%** | **19.41** | **23.80** | **15.42%** | **₹207,543** | **0.001s** |

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
- Python 3.10+
- Key dependencies: streamlit, pandas, 
umpy, xgboost, 	orch, plotly, scikit-learn, scipy, equests, pytest

### 2. Clone and Setup
`ash
git clone https://github.com/your-username/supply_demand_forecast_ai.git
cd supply_demand_forecast_ai
pip install -r requirements.txt
`

### 3. Launch the Interactive Dashboard
`ash
streamlit run app.py
`
*Access the app in your browser at http://localhost:8501*

### 4. Run Automated Unit Tests
`ash
python -m pytest tests/test_scout.py
`
*Expected output: 5 passed in ~5.2s (100% pass rate)*

---

## 🛠️ Project Structure

`
supply_demand_forecast_ai/
├── app.py                      # Main Streamlit Executive Dashboard
├── README.md                   # System Documentation
├── requirements.txt            # Python Dependencies
├── src/
│   ├── config.py               # API Keys, Currency, and Simulation Constants
│   ├── data/
│   │   └── generator.py        # Multi-Echelon Data Warehouse Generator & Cache
│   ├── forecasting/
│   │   ├── models.py           # Moving Avg, Seasonal, XGBoost & PyTorch LSTM
│   │   └── evaluator.py        # Model Evaluator & WAPE/Business Cost Metrics
│   ├── inventory/
│   │   └── risk_engine.py      # Dynamic Safety Stock, ROP & Stockout Probability
│   ├── suppliers/
│   │   └── supplier_engine.py  # Supplier Risk Scorecards & Bottleneck Propagation
│   ├── decision/
│   │   ├── recommender.py      # Actionable Priority Recommendations Engine
│   │   └── simulator.py        # What-If Scenario Stress Testing Simulator
│   ├── copilot/
│   │   └── llm_client.py       # Multi-Provider LLM Copilot (Groq/Gemini/Mistral)
│   └── ui/
│       ├── styles.py           # Custom Glassmorphic Dark Executive CSS
│       └── components.py       # Plotly Interactive Charts & Heatmaps
└── tests/
    └── test_scout.py           # Pytest Test Suite
`

---

## 📄 License
Distributed under the MIT License. See LICENSE for more information.
