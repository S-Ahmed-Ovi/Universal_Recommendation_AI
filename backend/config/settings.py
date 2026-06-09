"""
Global configuration and environment settings.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
from colorama import init

# NumPy compatibility patch for older aliases.
if not hasattr(np, "NaN"):
    np.NaN = np.nan

warnings.filterwarnings("ignore")
init(autoreset=True)

BASE_DIR = Path(__file__).resolve().parents[1]

# Data splitting
TEST_RATIO = float(os.getenv("TEST_RATIO", "0.25"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# Recommendation
TOP_K = int(os.getenv("TOP_K", "10"))
TOP_K_ALLOWED = (5, 10)

# Benchmark / scale limits
MAX_DENSE_ELEMENTS = int(os.getenv("MAX_DENSE_ELEMENTS", "500000000"))  # ~2GB float32
MIN_USER_INTERACTIONS = int(os.getenv("MIN_USER_INTERACTIONS", "1"))
MIN_ITEM_INTERACTIONS = int(os.getenv("MIN_ITEM_INTERACTIONS", "1"))

# Auto-downsample tiers
TIER_A_USERS = int(os.getenv("TIER_A_USERS", "10000"))
TIER_A_ITEMS = int(os.getenv("TIER_A_ITEMS", "5000"))
TIER_B_USERS = int(os.getenv("TIER_B_USERS", "3000"))
TIER_B_ITEMS = int(os.getenv("TIER_B_ITEMS", "2000"))

# Optuna
OPTUNA_TRIALS_AUTO = -1  # -1 means auto budget based on algorithm complexity
OPTUNA_TIMEOUT_S = int(os.getenv("OPTUNA_TIMEOUT_S", "600"))

# Model storage
MODEL_STORE_DIR = str((BASE_DIR / os.getenv("MODEL_STORE_DIR", "model_store")).resolve())

# Optuna SQLite persistence
OPTUNA_SQLITE_PATH = str((BASE_DIR / os.getenv("OPTUNA_SQLITE_PATH", "results/optuna_studies.db")).resolve())
OPTUNA_SQLITE_URL = f"sqlite:///{OPTUNA_SQLITE_PATH}" if OPTUNA_SQLITE_PATH else None

# Top model selection (ranked recommendations of algorithms/models)
TOP_N_MODELS = int(os.getenv("TOP_N_MODELS", "10"))
TOP_MODEL_ALLOWED = (5, 10)

# Algorithm mode
DEFAULT_ALGORITHM_MODE = os.getenv("DEFAULT_ALGORITHM_MODE", "auto").lower()
ALGORITHM_MODES = ("explicit", "implicit", "hybrid", "auto")

# API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Authentication
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "admin")      # default → "admin"
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "admin123")
AUTH_TOKEN_TTL_MINUTES = int(os.getenv("AUTH_TOKEN_TTL_MINUTES", "120"))
AUTH_SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "replace-me-in-production")
USERS_FILE = str((BASE_DIR / os.getenv("USERS_FILE", "users.json")).resolve())

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = str((BASE_DIR / os.getenv("LOG_FILE", "logs/proactive_ai.log")).resolve())

# Multi-tenancy
BASE_TENANTS_DIR = str((BASE_DIR / os.getenv("BASE_TENANTS_DIR", "tenants")).resolve())

# Feature flags (v2 → v3 migration toggles)
USE_V3_AUTH          = os.getenv("USE_V3_AUTH",          "true").lower() == "true"
USE_V3_MULTITENANCY  = os.getenv("USE_V3_MULTITENANCY",  "true").lower() == "true"
USE_V3_METERING      = os.getenv("USE_V3_METERING",      "true").lower() == "true"
USE_V3_ROUTES        = os.getenv("USE_V3_ROUTES",        "true").lower() == "true"
ALLOW_INACTIVE_MODEL_SERVING = os.getenv("ALLOW_INACTIVE_MODEL_SERVING", "false").lower() == "true"

# LLM Explanations
LLM_EXPLANATIONS_ENABLED = os.getenv("LLM_EXPLANATIONS_ENABLED", "false").lower() == "true"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower() # openai, local, groq
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", None) # For local providers like Ollama/LM Studio

TENANTS_DIR=str((BASE_DIR / os.getenv("TENANTS_DIR", "tenants")).resolve())
