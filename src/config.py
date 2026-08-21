import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
CACHE_DIR = DATA_DIR / 'cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# API Keys configured for SCOUT AI Supply Chain Copilot
# (Loaded from environment variables or .env file)
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
MISTRAL_API_KEY = os.environ.get('MISTRAL_API_KEY', '')
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')

# Supply Chain Simulation Constants
CURRENCY_SYMBOL = '₹'
DEFAULT_SERVICE_LEVEL = 0.95  # Z-score ~ 1.645
EXPEDITE_COST_PER_UNIT = 85.0 # Extra cost per unit for express logistics
