"""Refiner Agent — post-processes predictions for improved accuracy."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RefinementResult:
    refined_predictions: np.ndarray
    raw_predictions: np.ndarray
    method: str = "none"
    improvement_pct: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    # Judgmental override support (Hyndman: judgmental forecasting)
    manual_overrides: list[dict] = field(default_factory=list)
    override_audit: list[str] = field(default_factory=list)


class Refiner:
    """Post-processes raw predictions for enhanced accuracy.

    Supports:
    - Statistical refinement (bias correction, threshold optimization)
    - Judgmental overrides: manually adjust specific predictions with
      audit trail (per Hyndman's judgmental forecasting methodology)

    Usage:
        >>> refiner = Refiner()
        >>> result = refiner.refine(raw_pred, task_type="regression", method="statistical")
        >>> result = refiner.apply_override(result, {5: 342.0}, reason="Market rule change")
    """

    def __init__(self) -> None:
        self._methods = {
            "none": self._noop,
            "threshold_optimization": self._threshold_opt,
            "residual_correction": self._residual_correct,
            "statistical": self._statistical,
        }

    def refine(
        self, raw: np.ndarray, task_type: str = "regression",
        method: str = "none", reference: Optional[np.ndarray] = None, **kw: Any,
    ) -> RefinementResult:
        """Apply statistical refinement. For judgmental overrides, use apply_override."""
        fn = self._methods.get(method, self._noop)
        refined, details = fn(raw, task_type, reference, **kw)
        improvement = self._compute_improvement(raw, refined)
        return RefinementResult(
            refined_predictions=refined, raw_predictions=raw,
            method=method, improvement_pct=round(improvement * 100, 2), details=details,
        )

    def apply_override(
        self,
        result: RefinementResult,
        overrides: dict[int, float],
        reason: str = "",
        operator: str = "",
    ) -> RefinementResult:
        """Apply manual judgmental overrides with full audit trail.

        Per Hyndman's forecasting methodology: human experts possess
        information not available to models (policy changes, market
        rumors, weather events). This method records exactly what was
        changed, by whom, and why — essential for audit compliance.

        Args:
            result: A RefinementResult to modify.
            overrides: Dict of {index: new_value} overrides.
            reason: Why the override was applied (audit trail).
            operator: Who applied the override (audit trail).

        Returns:
            RefinementResult with overrides applied and audit trail.
        """
        import json
        from datetime import datetime, timezone

        modified = result.refined_predictions.copy()
        records = list(result.manual_overrides)
        audit = list(result.override_audit)

        for idx, new_val in overrides.items():
            if 0 <= idx < len(modified):
                old_val = float(modified[idx])
                modified[idx] = new_val
                records.append({
                    "index": int(idx),
                    "original": round(old_val, 4),
                    "override": round(new_val, 4),
                    "delta": round(new_val - old_val, 4),
                })
                timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
                audit.append(
                    f"[{timestamp}] [{operator or 'user'}] "
                    f"index={idx}: {old_val:.4f} → {new_val:.4f} "
                    f"(Δ={new_val - old_val:+.4f}) | reason: {reason}"
                )

        return RefinementResult(
            refined_predictions=modified,
            raw_predictions=result.raw_predictions,
            method=f"{result.method}+judgmental",
            improvement_pct=result.improvement_pct,
            details={**result.details, "judgmental_overrides_applied": len(records)},
            manual_overrides=records,
            override_audit=audit,
        )

    def _noop(self, pred, task, ref=None, **kw):
        return pred, {"method": "none"}

    def _threshold_opt(self, pred, task, ref=None, **kw):
        if task != "classification" or ref is None:
            return pred, {"applied": False}
        best_f1, best_t = 0.0, 0.5
        for t in np.linspace(0.1, 0.9, 40):
            b = (pred >= t).astype(int)
            tp, fp, fn = np.sum((b == 1) & (ref == 1)), np.sum((b == 1) & (ref == 0)), np.sum((b == 0) & (ref == 1))
            p = tp / (tp + fp) if (tp + fp) else 0
            r = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2 * p * r / (p + r) if (p + r) else 0
            if f1 > best_f1: best_f1, best_t = f1, t
        return (pred >= best_t).astype(float), {
            "applied": True,
            "method": "threshold_optimization",
            "optimized_threshold": round(best_t, 3),
            "correction_formula": f"(raw_pred >= {round(best_t, 3)}).astype(float)",
        }

    def _residual_correct(self, pred, task, ref=None, **kw):
        if ref is None: return pred, {"applied": False}
        bias = float(np.mean(ref - pred))
        return pred + bias, {
            "applied": True,
            "method": "residual_correction",
            "bias_correction": round(bias, 4),
            "correction_formula": f"raw_pred + ({round(bias, 4)})",
        }

    def _statistical(self, pred, task, ref=None, **kw):
        result = pred.copy()
        bias = 0.0
        if ref is not None:
            bias = float(np.mean(ref - pred))
            result = result + bias
        lo = float(np.percentile(pred, 1))
        hi = float(np.percentile(pred, 99))
        result = np.clip(result, lo, hi)
        return result, {
            "applied": True,
            "method": "statistical",
            "bias_correction": round(bias, 4),
            "clip_lower": round(lo, 4),
            "clip_upper": round(hi, 4),
            "correction_formula": (
                f"clip(raw_pred + ({round(bias, 4)}), "
                f"{round(lo, 4)}, {round(hi, 4)})"
            ),
        }

    @staticmethod
    def _compute_improvement(raw: np.ndarray, refined: np.ndarray) -> float:
        if np.allclose(raw, refined): return 0.0
        rv = float(np.var(raw)) or 1e-10
        return max(0.0, 1.0 - float(np.var(raw - refined)) / rv)