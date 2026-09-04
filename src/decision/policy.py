"""
Phase 5: Three-way operational decision policy (APPROVE / REVIEW / BLOCK).

A fraud probability is a ranking signal, not an operational decision by
itself — someone still has to decide what score is "low enough to let
through automatically" and what score is "high enough to block
automatically," and those choices trade off fraud losses against
legitimate-customer friction and finite manual-review capacity. This
module implements that conversion as an explicit, reusable, testable
policy object rather than inline threshold comparisons scattered through
scripts.

Score direction (explicit, matching every other score in this project):
HIGHER score = HIGHER fraud risk.

Decision rule (exactly the brief's pseudocode):
    if risk_score < approve_threshold:  APPROVE
    elif risk_score < block_threshold:  REVIEW
    else:                                BLOCK

Boundary behavior is therefore: a score exactly equal to `approve_threshold`
falls into REVIEW (not APPROVE — the comparison is strict `<`), and a score
exactly equal to `block_threshold` falls into BLOCK (the final `else`
catches `score >= block_threshold`). This is a deliberate, testable
convention, not an accident of implementation.

These are modeling-policy decisions, evaluated against historical
validation/test data — not guaranteed real-world financial rules, legal
requirements, or a claim that this exact policy is ready for production
deployment. See the Phase 5 report for the full set of caveats.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DECISIONS = ("APPROVE", "REVIEW", "BLOCK")


class InvalidThresholdError(ValueError):
    pass


@dataclass
class DecisionPolicy:
    approve_threshold: float
    block_threshold: float
    name: str = "unnamed_policy"

    def __post_init__(self):
        for label, value in (("approve_threshold", self.approve_threshold), ("block_threshold", self.block_threshold)):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise InvalidThresholdError(f"{label} must be numeric, got {type(value)}")
            if not (0.0 <= value <= 1.0):
                raise InvalidThresholdError(f"{label} must be in [0, 1], got {value}")
        if self.approve_threshold > self.block_threshold:
            raise InvalidThresholdError(
                f"approve_threshold ({self.approve_threshold}) must be <= "
                f"block_threshold ({self.block_threshold}) — thresholds are out of order."
            )

    def decide(self, scores) -> np.ndarray:
        """
        Returns an array of decision strings, one per input score, drawn
        from `DECISIONS`. Every transaction receives exactly one decision
        (no NaN/undecided category is possible by construction — every
        real number falls into exactly one of the three branches).
        """
        scores = np.asarray(scores, dtype="float64")
        if np.isnan(scores).any():
            raise ValueError(
                "DecisionPolicy.decide received NaN score(s); a fraud "
                "probability must be a real number for every transaction "
                "before a decision can be made."
            )
        decisions = np.full(scores.shape, "REVIEW", dtype=object)
        decisions[scores < self.approve_threshold] = "APPROVE"
        decisions[scores >= self.block_threshold] = "BLOCK"
        return decisions

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "approve_threshold": self.approve_threshold,
            "block_threshold": self.block_threshold,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionPolicy":
        return cls(
            approve_threshold=d["approve_threshold"],
            block_threshold=d["block_threshold"],
            name=d.get("name", "unnamed_policy"),
        )
