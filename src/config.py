import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
CACHE_DIR = DATA_DIR / 'cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_api_key(key_name: str) -> str:
    # 1. Check Streamlit Cloud Secrets (st.secrets)
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if key_name in st.secrets:
                val = str(st.secrets[key_name]).strip()
                if val: return val
            if key_name.lower() in st.secrets:
                val = str(st.secrets[key_name.lower()]).strip()
                if val: return val
    except Exception:
        pass

    # 2. Check Environment variables
    env_val = os.environ.get(key_name, "").strip()
    if env_val:
        return env_val

    return ""

GROQ_API_KEY = get_api_key("GROQ_API_KEY")
GEMINI_API_KEY = get_api_key("GEMINI_API_KEY")
MISTRAL_API_KEY = get_api_key("MISTRAL_API_KEY")
OPENROUTER_API_KEY = get_api_key("OPENROUTER_API_KEY")

# Supply Chain Simulation Constants
CURRENCY_SYMBOL = '₹'
DEFAULT_SERVICE_LEVEL = 0.95  # Z-score ~ 1.645
EXPEDITE_COST_PER_UNIT = 85.0 # Extra cost per unit for express logistics
