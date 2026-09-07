"""
Production upgrade — SHAP explainability.

STRICTLY presentation-only. `FraudExplainer.explain()` is called AFTER
`RiskDecisionEngine.process_transaction()` has already computed
`risk_score` and `decision` from the real model and the real frozen
policy — it takes the EXACT SAME preprocessed feature matrix (`Z_full`)
that produced that real score, and explains THAT matrix. There is no
path by which this module can change a risk score or a decision:

    Prediction   (LightGBM predict_proba)   -> risk_score      [unchanged]
    Decision     (frozen Phase 5 policy)     -> decision         [unchanged]
    Explanation  (SHAP TreeExplainer)          -> top_features     [ADDITIVE ONLY]

Uses `shap.TreeExplainer`, which computes exact Shapley values for
tree-ensemble models (LightGBM's own boosted-tree structure) — not an
approximation requiring background sampling, which is what makes it fast
enough to consider for the synchronous request path in the first place
(see the real, measured latency numbers in
`reports/streaming_benchmark.md`).
"""

from __future__ import annotations


class FraudExplainer:
    def __init__(self, model, top_k: int = 5):
        import shap
        self._shap = shap.TreeExplainer(model)
        self.top_k = top_k

    def explain(self, Z_full) -> dict:
        """
        `Z_full` must be the exact preprocessed, single-row feature
        DataFrame already used for the real prediction — never
        recomputed independently by this method.
        """
        raw_shap_values = self._shap.shap_values(Z_full)

        # shap's TreeExplainer output shape for a binary LightGBM
        # classifier has varied across shap versions (a list of two
        # per-class arrays vs. a single 2D array) — handle both without
        # guessing which the installed version returns.
        if isinstance(raw_shap_values, list):
            values = raw_shap_values[1][0]  # class-1 (fraud) contributions, first (only) row
            base_value = self._shap.expected_value[1] if isinstance(self._shap.expected_value, (list, tuple)) else self._shap.expected_value
        else:
            values = raw_shap_values[0]
            if values.ndim > 1:  # some versions return (n_rows, n_features, n_classes)
                values = values[:, 1]
            base_value = self._shap.expected_value
            if isinstance(base_value, (list, tuple)):
                base_value = base_value[1]

        feature_names = list(Z_full.columns)
        pairs = list(zip(feature_names, [float(v) for v in values]))
        pairs.sort(key=lambda p: abs(p[1]), reverse=True)
        top = pairs[: self.top_k]

        return {
            "top_features": [{"feature": name, "impact": impact} for name, impact in top],
            "base_value": float(base_value) if base_value is not None else None,
        }
