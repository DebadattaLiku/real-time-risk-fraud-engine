#!/usr/bin/env python3
"""
Production upgrade — log the existing, already-approved champion into
MLflow (real metrics, real hyperparameters, real model artifact — see
src/mlops/mlflow_tracking.py's module docstring for what this is and
is NOT: a registration of an existing model, never a new training run).

Usage:
    python scripts/log_champion_to_mlflow.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mlops.mlflow_tracking import log_champion_to_mlflow


def main() -> int:
    run_id = log_champion_to_mlflow()
    print(f"Logged the real champion (fraud-risk-lightgbm-v1) to MLflow. Run ID: {run_id}")
    print("View with: mlflow ui --backend-store-uri sqlite:///mlflow.db")
    return 0


if __name__ == "__main__":
    sys.exit(main())
