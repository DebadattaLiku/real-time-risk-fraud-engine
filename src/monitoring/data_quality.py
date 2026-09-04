"""
Phase 9: Data-quality signal extraction.

Pure functions that pull SAFE, aggregate-only signals out of a transaction
dict for monitoring purposes — never the raw payload itself, never
anything that could identify a specific transaction beyond what's already
in the (already-approved, Phase 7) API response. These are read-only
inspections; nothing here mutates the transaction dict passed in.
"""

from __future__ import annotations


def extract_transaction_amount(transaction: dict, amount_col: str = "TransactionAmt") -> float | None:
    """Safe, single-number signal for monitoring — not the full payload."""
    value = transaction.get(amount_col)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def count_missing_optional_fields(transaction: dict, required_fields: set) -> int:
    """
    How many of the OPTIONAL (non-required) fields in this transaction are
    missing/None. A coarse, aggregate-only data-quality signal — does not
    reveal which specific fields, still less their values.
    """
    return sum(
        1 for key, value in transaction.items()
        if key not in required_fields and value is None
    )
