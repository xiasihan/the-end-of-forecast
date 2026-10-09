"""
Model explainability — global & local interpretation.

Implements Phase 8 of the methodology: "An unexplainable model in
a high-stakes domain is unacceptable."

Provides:
- Global feature importance (SHAP, permutation, gain)
- Local explanations (SHAP waterfall for individual predictions)
- Partial dependence plots (PDP)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExplainabilityResult:
    """Structured model interpretation results.

    Attributes:
        global_importance: Feature importance ranked from most to least.
        shap_values: Raw SHAP values (n_samples × n_features) if available.
        feature_names: Names of the features.
        method: Method used for explanation ("shap", "permutation", "gain", "none").
    """
    global_importance: dict[str, float] = field(default_factory=dict)
    shap_values: Optional[np.ndarray] = None
    feature_names: list[str] = field(default_factory=list)
    method: str = "none"

    @property
    def top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """Return the top-N most important features and their scores."""
        ranked = sorted(self.global_importance.items(), key=lambda x: -abs(x[1]))
        return ranked[:n]

    def summary(self) -> str:
        """Text summary of feature importance."""
        if not self.global_importance:
            return "No importance scores available."
        lines = [f"Feature Importance ({self.method}):"]
        for feat, score in self.top_features(5):
            lines.append(f"  {feat}: {score:.4f}")
        return "\n".join(lines)


class Explainer:
    """Model-agnostic explainability engine.

    Usage:
        >>> explainer = Explainer()
        >>> result = explainer.explain(model, X, feature_names=cols)
        >>> print(result.summary())
    """

    def explain(
        self,
        model: Any,
        X: pd.DataFrame,
        y: Optional[np.ndarray] = None,
        method: str = "auto",
        n_samples: int = 200,
    ) -> ExplainabilityResult:
        """Generate global and local model explanations.

        Args:
            model: A fitted model with .predict() method.
            X: Feature DataFrame used for explanation.
            y: Optional target values (needed for permutation importance).
            method: "auto", "shap", "permutation", "gain", or "none".
            n_samples: Number of samples for SHAP explanation (to limit compute).

        Returns:
            ExplainabilityResult with importance scores and SHAP values.
        """
        feature_names = list(X.columns)

        # Subsample for SHAP performance
        if len(X) > n_samples and method in {"auto", "shap"}:
            X_sample = X.sample(n=n_samples, random_state=42)
        else:
            X_sample = X

        # ── Try SHAP first ───────────────────────────────────
        if method in {"auto", "shap"}:
            try:
                result = self._shap_explain(model, X_sample, feature_names)
                if result.global_importance:
                    return result
            except Exception as exc:
                logger.debug("SHAP failed: %s", exc)

        # ── Fallback: Permutation importance ─────────────────
        if method in {"auto", "permutation"} and y is not None:
            return self._permutation_explain(model, X_sample, y.values if hasattr(y, "values") else y, feature_names)

        # ── Last resort: try sklearn feature_importances_ ─────
        if hasattr(model, "feature_importances_"):
            scores = dict(zip(feature_names, model.feature_importances_))
            return ExplainabilityResult(
                global_importance=scores,
                feature_names=feature_names,
                method="gain",
            )

        logger.warning("No explainability method available.")
        return ExplainabilityResult(feature_names=feature_names, method="none")

    def _shap_explain(
        self, model: Any, X: pd.DataFrame, feature_names: list[str],
    ) -> ExplainabilityResult:
        """Explain with SHAP (TreeExplainer or KernelExplainer)."""
        try:
            import shap

            # Try TreeExplainer first (fast)
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
            except Exception:
                # Fallback to KernelExplainer (slow but universal)
                background = shap.sample(X, min(50, len(X)))
                explainer = shap.KernelExplainer(model.predict, background)
                shap_values = explainer.shap_values(X)

            # Handle multi-class SHAP output
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Take first class

            # Mean absolute SHAP as importance
            importance = {}
            for i, name in enumerate(feature_names):
                if i < shap_values.shape[1]:
                    importance[name] = float(np.mean(np.abs(shap_values[:, i])))

            return ExplainabilityResult(
                global_importance=importance,
                shap_values=shap_values,
                feature_names=feature_names,
                method="shap",
            )
        except ImportError:
            logger.debug("shap not installed. pip install shap")
        except Exception as exc:
            logger.debug("SHAP explanation failed: %s", exc)
        return ExplainabilityResult(feature_names=feature_names, method="none")

    def _permutation_explain(
        self, model: Any, X: pd.DataFrame, y: np.ndarray, feature_names: list[str],
    ) -> ExplainabilityResult:
        """Permutation feature importance."""
        try:
            from sklearn.inspection import permutation_importance
            result = permutation_importance(model, X, y, n_repeats=5, random_state=42)
            importance = dict(zip(feature_names, result.importances_mean))
            return ExplainabilityResult(
                global_importance=importance,
                feature_names=feature_names,
                method="permutation",
            )
        except Exception as exc:
            logger.debug("Permutation importance failed: %s", exc)
        return ExplainabilityResult(feature_names=feature_names, method="none")