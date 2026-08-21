import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, List

def create_forecast_chart(
    hist_dates: List[Any],
    hist_demand: List[float],
    future_dates: List[Any],
    p10: List[float],
    p50: List[float],
    p90: List[float],
    sku_name: str,
    warehouse_name: str,
    model_name: str = "XGBoost Regressor"
) -> go.Figure:
    fig = go.Figure()

    # Historical demand line
    fig.add_trace(go.Scatter(
        x=hist_dates,
        y=hist_demand,
        mode="lines+markers",
        name="Historical Actual Demand",
        line=dict(color="#38BDF8", width=2.5),
        marker=dict(size=4)
    ))

    # P90 Upper Bound (for envelope)
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=p90,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        name="Upper Bound (P90)"
    ))

    # P10 Lower Bound + Fill between P10 and P90
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=p10,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(99, 102, 241, 0.22)",
        name="90% Confidence Interval (P10 - P90)"
    ))

    # P50 Forecast Median
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=p50,
        mode="lines+markers",
        name=f"Forecast ({model_name})",
        line=dict(color="#818CF8", width=3, dash="dash"),
        marker=dict(size=6, color="#C7D2FE")
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>Probabilistic Demand Forecast</b>  {sku_name} @ {warehouse_name}",
            font=dict(size=16, color="#F3F4F6")
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(17, 24, 39, 0.4)",
        plot_bgcolor="rgba(17, 24, 39, 0.6)",
        hovermode="x unified",
        margin=dict(l=40, r=30, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11)
        ),
        xaxis=dict(gridcolor="rgba(255, 255, 255, 0.08)", title="Date"),
        yaxis=dict(gridcolor="rgba(255, 255, 255, 0.08)", title="Daily Units Demand")
    )
    return fig

def create_stockout_heatmap(inventory_df: pd.DataFrame) -> go.Figure:
    pivot_df = inventory_df.pivot_table(
        index="category",
        columns="warehouse_id",
        values="stockout_probability_pct",
        aggfunc="mean"
    ).fillna(0)

    fig = px.imshow(
        pivot_df,
        labels=dict(x="Warehouse", y="Product Category", color="Stockout Risk %"),
        x=pivot_df.columns,
        y=pivot_df.index,
        color_continuous_scale=[[0, "#10B981"], [0.4, "#FBBF24"], [0.7, "#F97316"], [1.0, "#EF4444"]],
        aspect="auto"
    )

    fig.update_layout(
        title=dict(
            text="<b>Multi-Warehouse Stockout Risk Heatmap (%)</b>",
            font=dict(size=15, color="#F3F4F6")
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(17, 24, 39, 0.4)",
        plot_bgcolor="rgba(17, 24, 39, 0.6)",
        margin=dict(l=40, r=30, t=50, b=40)
    )
    return fig

def create_supplier_radar_chart(suppliers_df: pd.DataFrame) -> go.Figure:
    top_suppliers = suppliers_df.head(6)
    categories = ["OTIF Reliability %", "Quality Score (x20)", "Low Delay Rate %", "Lead Time Speed %"]
    
    fig = go.Figure()
    colors = ["#6366F1", "#EC4899", "#10B981", "#F59E0B", "#06B6D4", "#8B5CF6"]

    for i, (_, sup) in enumerate(top_suppliers.iterrows()):
        otif = sup["otif_reliability_pct"]
        quality = sup["quality_score"] * 20.0
        low_delay = max(0, 100.0 - sup["historical_delay_rate_pct"])
        speed = max(0, 100.0 - (sup["avg_lead_time_days"] / 14.0 * 100.0))
        
        values = [otif, quality, low_delay, speed]
        values.append(values[0])
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            name=sup["supplier_name"][:16],
            line=dict(color=colors[i % len(colors)], width=2),
            opacity=0.65
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color="#9CA3AF", gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(color="#E5E7EB", gridcolor="rgba(255,255,255,0.1)")
        ),
        title=dict(
            text="<b>Top Supplier Multi-Dimensional Capability Radar</b>",
            font=dict(size=15, color="#F3F4F6")
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(17, 24, 39, 0.4)",
        plot_bgcolor="rgba(17, 24, 39, 0.6)",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", y=-0.15)
    )
    return fig

def create_simulation_impact_chart(sim_results: Dict[str, Any]) -> go.Figure:
    categories = [c["category"] for c in sim_results["category_impact"]]
    lost_rev = [c["total_lost_revenue"] / 100000.0 for c in sim_results["category_impact"]] # in Lakhs
    stockouts = [c["stockout_items"] for c in sim_results["category_impact"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories,
        y=lost_rev,
        name="Estimated Revenue At Risk (? Lakhs)",
        marker=dict(color="#EF4444", opacity=0.85),
        yaxis="y"
    ))
    fig.add_trace(go.Scatter(
        x=categories,
        y=stockouts,
        name="Stockout SKU Count",
        mode="lines+markers",
        line=dict(color="#FBBF24", width=3),
        marker=dict(size=8),
        yaxis="y2"
    ))

    fig.update_layout(
        title=dict(
            text="<b>What-If Impact Breakdown by Product Category</b>",
            font=dict(size=15, color="#F3F4F6")
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(17, 24, 39, 0.4)",
        plot_bgcolor="rgba(17, 24, 39, 0.6)",
        yaxis=dict(title="Lost Revenue (? Lakhs)", gridcolor="rgba(255,255,255,0.08)"),
        yaxis2=dict(
            title="Stockout SKUs",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center")
    )
    return fig

def create_inventory_doi_donut(inventory_df: pd.DataFrame) -> go.Figure:
    status_counts = inventory_df["status_label"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]
    
    color_map = {
        "Stocked Out": "#EF4444",
        "High Stockout Risk": "#F97316",
        "Reorder Needed": "#FBBF24",
        "Optimal": "#10B981",
        "Excess Inventory": "#3B82F6"
    }

    fig = px.pie(
        status_counts,
        names="Status",
        values="Count",
        hole=0.55,
        color="Status",
        color_discrete_map=color_map
    )

    fig.update_layout(
        title=dict(
            text="<b>Network Inventory Health Breakdown</b>",
            font=dict(size=15, color="#F3F4F6")
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(17, 24, 39, 0.4)",
        plot_bgcolor="rgba(17, 24, 39, 0.6)",
        margin=dict(l=30, r=30, t=50, b=30),
        legend=dict(orientation="h", y=-0.1)
    )
    return fig
