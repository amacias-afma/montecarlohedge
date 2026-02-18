"""
Project configuration.

This module centralizes all path and parameter configurations.
Users should update DATA_PATH to point to their local data directory.
"""
import os

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Raw data directory: update this to your local data path
# Default assumes data is in a sibling 'Data' folder or the user's Data directory
DATA_PATH = os.environ.get("MCHEDGE_DATA_PATH", "C:/Users/fe_ma/Data/")

# Project internal paths
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "cache")
ARTICLE_DIR = os.path.join(PROJECT_ROOT, "article")
ARTICLE_FIGURES_DIR = os.path.join(ARTICLE_DIR, "figures")

# Ensure output directories exist
for d in [RESULTS_DIR, FIGURES_DIR, TABLES_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)
