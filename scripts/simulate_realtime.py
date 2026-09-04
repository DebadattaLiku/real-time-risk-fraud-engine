#!/usr/bin/env python3
"""
Phase 6 demonstration script — real-time transaction simulation.

Loads the trained model bundle (building and caching it on first run if
not already present — no internal source edits required), warm-starts
behavioral state from historical data, then processes a chronological
sequence of transactions one at a time through `RiskDecisionEngine`,
printing each decision as it happens.

This is a LOCAL SIMULATION for demonstration and validation purposes — see
`reports/phase6_realtime_engine_summary.md` for the distinction between
this and an actual deployed, production real-time system.

Usage (from the repository root):

    python scripts/simulate_realtime.py
    python scripts/simulate_realtime.py --n-transactions 20
    python scripts/simulate_realtime.py --n-transactions 10 --quiet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import yaml

from src.data.load import load_train_transaction, load_config
from src.data.split import compute_temporal_split
from src.decision.policy import DecisionPolicy
from src.engine.state import BehavioralStateManager
from src.engine.risk_engine import RiskDecisionEngine
from src.run_phase6_simulation import build_or_load_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate real-time transaction processing.")
    parser.add_argument("--n-transactions", type=int, default=15,
                         help="Number of test transactions to process one-by-one (default: 15)")
    parser.add_argument("--quiet", action="store_true", help="Only print the final summary")
    args = parser.parse_args()

    config = load_config()
    id_col, time_col, target_col, entity_col = "TransactionID", "TransactionDT", "isFraud", "card1"

    print("Loading model bundle (training + caching on first run if needed)...")
    bundle = build_or_load_bundle(config)
    schema, fp, lgbm_pre, model = bundle["schema"], bundle["feature_pipeline"], bundle["lgbm_preprocessor"], bundle["model"]

    print("Loading data and frozen decision policy...")
    df, _ = load_train_transaction(config=config)
    train_df, val_df, test_df, _ = compute_temporal_split(
        df, time_col, id_col, config["split"]["train_ratio"], config["split"]["val_ratio"], config["split"]["test_ratio"],
    )
    del df

    with open(REPO_ROOT / "config" / "decision_policy.yaml") as f:
        policy_config = yaml.safe_load(f)
    policy = DecisionPolicy(
        approve_threshold=policy_config["approve_threshold"],
        block_threshold=policy_config["block_threshold"],
        name=policy_config["policy_name"],
    )

    print("Warm-starting behavioral state from historical (train+validation) data...")
    historical = pd.concat([train_df, val_df], ignore_index=True)
    state_manager = BehavioralStateManager(entity_col=entity_col)
    state_manager.bulk_initialize(historical, time_col=time_col, amount_col="TransactionAmt",
                                   as_of_time=float(historical[time_col].max()))

    engine = RiskDecisionEngine(
        schema=schema, feature_pipeline=fp, lgbm_preprocessor=lgbm_pre, model=model,
        policy=policy, state_manager=state_manager, entity_col=entity_col, time_col=time_col,
        id_col=id_col, amount_col="TransactionAmt",
    )

    test_sorted = test_df.sort_values([time_col, id_col], kind="mergesort").reset_index(drop=True)
    sim_slice = test_sorted.iloc[:args.n_transactions]

    print(f"\nProcessing {len(sim_slice)} transactions one at a time "
          f"(chronological order, deterministic (TransactionDT, TransactionID))...\n")
    print(f"{'TransactionID':>14} | {'RiskScore':>9} | {'Decision':<8} | {'PriorTxns':>9} | {'TimeMs':>6}")
    print("-" * 65)

    results = []
    for _, row in sim_slice.iterrows():
        txn = row.to_dict()
        txn.pop(target_col, None)  # isFraud is NEVER passed to the engine
        result = engine.process_transaction(txn)
        results.append(result)
        if not args.quiet:
            print(f"{result['transaction_id']:>14} | {result['risk_score']:>9.4f} | "
                  f"{result['decision']:<8} | {int(result['behavioral_features']['bhv_prev_txn_count']):>9} | "
                  f"{result['processing_metadata']['processing_time_ms']:>6.1f}")

    n_approve = sum(1 for r in results if r["decision"] == "APPROVE")
    n_review = sum(1 for r in results if r["decision"] == "REVIEW")
    n_block = sum(1 for r in results if r["decision"] == "BLOCK")
    avg_time = sum(r["processing_metadata"]["processing_time_ms"] for r in results) / len(results)

    print("\n" + "=" * 65)
    print(f"Summary: {len(results)} transactions processed")
    print(f"  APPROVE: {n_approve} ({n_approve/len(results):.1%})")
    print(f"  REVIEW:  {n_review} ({n_review/len(results):.1%})")
    print(f"  BLOCK:   {n_block} ({n_block/len(results):.1%})")
    print(f"  Average processing time: {avg_time:.1f} ms/transaction")
    print("\nThis is a LOCAL SIMULATION (no deployed API, no streaming "
          "infrastructure) — see reports/phase6_realtime_engine_summary.md.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
