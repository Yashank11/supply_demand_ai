import os
import json
import requests
from typing import Dict, Any, List, Optional
from src.config import GROQ_API_KEY, GEMINI_API_KEY, MISTRAL_API_KEY, OPENROUTER_API_KEY

class ScoutCopilotClient:
    """
    Intelligent AI Supply Chain Copilot with automated multi-provider fallback:
    Groq -> Gemini -> Mistral -> OpenRouter
    """
    def __init__(self):
        self.groq_key = GROQ_API_KEY
        self.gemini_key = GEMINI_API_KEY
        self.mistral_key = MISTRAL_API_KEY
        self.openrouter_key = OPENROUTER_API_KEY

    def _call_groq(self, prompt: str, system_context: str) -> Optional[str]:
        if not self.groq_key or "YOUR_" in self.groq_key:
            return None
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_context},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1500
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception:
            pass
        return None

    def _call_gemini(self, prompt: str, system_context: str) -> Optional[str]:
        if not self.gemini_key or "YOUR_" in self.gemini_key:
            return None
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.gemini_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"System Context:\n{system_context}\n\nUser Question:\n{prompt}"}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 2500
            }
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return "".join([p.get("text", "") for p in parts])
        except Exception:
            pass
        # Fallback to gemini-1.5-flash if 2.5 is unavailable
        try:
            url_15 = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_key}"
            resp = requests.post(url_15, headers=headers, json=payload, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return "".join([p.get("text", "") for p in parts])
        except Exception:
            pass
        return None

    def _call_mistral(self, prompt: str, system_context: str) -> Optional[str]:
        if not self.mistral_key or "YOUR_" in self.mistral_key:
            return None
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.mistral_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "mistral-large-latest",
            "messages": [
                {"role": "system", "content": system_context},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1500
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception:
            pass
        return None

    def _call_openrouter(self, prompt: str, system_context: str) -> Optional[str]:
        if not self.openrouter_key or "YOUR_" in self.openrouter_key:
            return None
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "meta-llama/llama-3.3-70b-instruct",
            "messages": [
                {"role": "system", "content": system_context},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1500
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception:
            pass
        return None

    def ask(self, prompt: str, live_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sends query to the AI Copilot with live supply chain RAG context and automatic fallback.
        """
        system_context = f"""
You are SCOUT, an elite Chief Supply Chain Intelligence AI Copilot and Operations Strategist.
You assist VPs of Supply Chain, Logistics Directors, and Inventory Planners in real-time.

Current Live Supply Chain Snapshot:
- Total Monitored SKUs: {live_context.get('total_skus', 40)}
- Active Warehouses: {live_context.get('total_warehouses', 10)}
- Critical Stockout Alerts: {live_context.get('critical_stockouts_count', 0)}
- Total Capital Locked in Inventory: ₹{live_context.get('total_capital_locked_inr', 0):,.0f}
- Estimated Total Revenue At Risk: ₹{live_context.get('total_revenue_at_risk_inr', 0):,.0f}

Top High-Risk SKUs:
{json.dumps(live_context.get('top_risk_skus', []), indent=2)}

Top Supplier Risk Scorecard:
{json.dumps(live_context.get('top_risk_suppliers', []), indent=2)}

Active Simulation State (if any):
{json.dumps(live_context.get('latest_simulation', {}), indent=2)}

Instructions for Your Response:
1. Deliver structured, decisive, and business-focused intelligence.
2. Quantify risks clearly (SKU IDs, Warehouses, Days to Stockout, Financial impact in ₹ Lakhs/Crores).
3. Offer actionable 3-step mitigation plans: Immediate (0-24h), Tactical (2-7 days), and Strategic (30+ days).
4. Maintain a sharp, executive, data-driven tone. Use markdown formatting with bullet points and bold highlights.
"""

        # Provider fallback chain
        providers = [
            ("Groq (Llama-3.3-70B)", self._call_groq),
            ("Google Gemini (2.5-Flash)", self._call_gemini),
            ("Mistral AI (Large)", self._call_mistral),
            ("OpenRouter (Llama-3.3-70B)", self._call_openrouter)
        ]
        
        for name, func in providers:
            try:
                ans = func(prompt, system_context)
                if ans and len(ans.strip()) > 20:
                    return {
                        "response": ans,
                        "provider_used": name,
                        "status": "success"
                    }
            except Exception:
                continue
                
        # Fallback offline heuristic engine if network/API quota is reached
        heuristic_reply = f"""
### ?? SCOUT Autonomous Fallback Intelligence Report

**Analysis for:** "{prompt}"

1. **Immediate Vulnerability Assessment**:
   - Current Critical Stockouts detected: **{live_context.get('critical_stockouts_count', 3)} SKUs**.
   - Total Revenue Exposure: **?{live_context.get('total_revenue_at_risk_inr', 1850000):,.0f}**.
   - Primary Bottleneck: Suppliers with high lead-time volatility and low OTIF rates.

2. **Immediate Action Plan**:
   - **Action 1 (Emergency Expedite)**: Expedite in-transit shipments for top risk SKUs using express air/courier logistics.
   - **Action 2 (Inter-Warehouse Rebalancing)**: Transfer buffer inventory from overstocked sister hubs (e.g. Pune -> Mumbai Central).
   - **Action 3 (Advance PO Placement)**: Trigger dynamic Reorder Point purchase orders with +20% safety margin.

*(Telemetry: Generated via SCOUT Offline Operational Heuristics Engine)*
"""
        return {
            "response": heuristic_reply,
            "provider_used": "SCOUT Heuristics Fallback Engine",
            "status": "fallback"
        }
