"""
Feature fingerprint extraction engine.

Computes statistical, distributional, and structural characteristics
from any dataset for intelligent method selection and similarity matching.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureFingerprint:
    """Compact data profile for method recommendation."""

    acf1: float = 0.0
    seasonal_strength: float = 0.0
    trend_strength: float = 0.0
    entropy: float = 0.0
    unitroot_stat: float = 0.0
    zero_ratio: float = 0.0
    extreme_ratio: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    n_series: int = 1
    n_obs: int = 0
    seasonality: int = 7
    freq: str = "D"
    missing_ratio: float = 0.0
    # Cold start / small-sample group detection (P1)
    min_group_size: int = 1
    groups_below_threshold: int = 0
    threshold_sample_size: int = 10
    tags: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tags:
            self.tags = {
                "is_sparse": self.zero_ratio > 0.3,
                "is_highly_seasonal": self.seasonal_strength > 0.5,
                "is_trended": self.trend_strength > 0.3,
                "is_short": self.n_obs < 500,
                "is_multi_series": self.n_series > 1,
                "has_missing": self.missing_ratio > 0.01,
                "is_cold_start": self.groups_below_threshold > 0,
            }

    def to_dict(self) -> dict:
        base = {k: v for k, v in self.__dict__.items() if k not in ("tags", "diagnostic")}
        base.update(self.tags)
        return base

    @property
    def diagnostic(self) -> str:
        """Generate structured natural-language diagnostic narrative.

        LLM agents need structured text descriptions, not isolated numbers.
        ACF1=0.85 alone doesn't convey 'slow decay with lag-7 bump'.
        The diagnostic text bridges this gap — it's the 'pseudo-visual'
        description that enables LLM-based model selection.
        """
        lines = []

        # ── Stationarity ──────────────────────────────────────
        if self.unitroot_stat < -3.0:
            lines.append(
                f"Stationarity: ADF statistic={self.unitroot_stat:.2f} — "
                "strongly rejects the unit-root hypothesis. "
                "The series is stationary; differencing is probably unnecessary."
            )
        elif self.unitroot_stat < -2.0:
            lines.append(
                f"Stationarity: ADF statistic={self.unitroot_stat:.2f} — "
                "weakly stationary. Differencing may help but is not essential."
            )
        elif self.n_obs > 2:
            lines.append(
                f"Stationarity: ADF statistic={self.unitroot_stat:.2f} — "
                "fails to reject the unit-root hypothesis. "
                "Strongly suggest first-order differencing (d=1 in ARIMA terms)."
            )

        # ── Trend ─────────────────────────────────────────────
        if self.trend_strength > 0.6:
            lines.append(
                f"Trend: strong (strength={self.trend_strength:.2f}). "
                "A clear monotonic trend dominates the series. "
                "Consider linear/multi-step trend modeling or differencing."
            )
        elif self.trend_strength > 0.3:
            lines.append(
                f"Trend: moderate ({self.trend_strength:.2f}). "
                "Detrending is advisable but not critical."
            )
        else:
            lines.append(
                f"Trend: weak/absent ({self.trend_strength:.2f}). "
                "Trend-based models are unlikely to help."
            )

        # ── Seasonality ───────────────────────────────────────
        if self.seasonal_strength > 0.5:
            lines.append(
                f"Seasonality: strong ({self.seasonal_strength:.2f}, period≈{self.seasonality}). "
                "A dominant seasonal pattern is present. "
                "Include seasonal differencing (D=1), Fourier terms, "
                "or explicit seasonal decomposition (STL). "
                "SeasonalNaive will be a strong baseline."
            )
        elif self.seasonal_strength > 0.2:
            lines.append(
                f"Seasonality: moderate ({self.seasonal_strength:.2f}). "
                "Weak seasonal signal — may help but don't over-weight it."
            )
        else:
            lines.append(
                f"Seasonality: negligible ({self.seasonal_strength:.2f}). "
                "No meaningful seasonal pattern detected. "
                "Skip seasonal models."
            )

        # ── Autocorrelation ───────────────────────────────────
        if abs(self.acf1) > 0.7:
            lines.append(
                f"Autocorrelation: very strong (ACF1={self.acf1:.3f}). "
                "The series is highly predictable from its own past. "
                "AR-type models (ARIMA, linear regression with lags) "
                "should perform well. Lag features are essential."
            )
        elif abs(self.acf1) > 0.3:
            lines.append(
                f"Autocorrelation: moderate (ACF1={self.acf1:.3f}). "
                "Some temporal dependence exists; lag features will help."
            )
        else:
            lines.append(
                f"Autocorrelation: weak (ACF1={self.acf1:.3f}). "
                "The series has little memory — tree/ML models with "
                "exogenous features may outperform pure time-series models."
            )

        # ── Distribution ──────────────────────────────────────
        if abs(self.skewness) > 1.5:
            direction = "right" if self.skewness > 0 else "left"
            lines.append(
                f"Distribution: heavily {direction}-skewed (skew={self.skewness:.2f}). "
                "Strongly recommend Box-Cox or log transformation before modeling."
            )
        elif abs(self.skewness) > 0.5:
            lines.append(
                f"Distribution: moderately skewed (skew={self.skewness:.2f}). "
                "Consider power transform or robust scaling."
            )

        if self.zero_ratio > 0.3:
            lines.append(
                f"Sparsity: {self.zero_ratio:.0%} of values are zero. "
                "Consider zero-inflated models or two-stage approaches "
                "(classify zero vs. non-zero, then regress non-zero values)."
            )

        # ── Overall recommendation ────────────────────────────
        lines.append(f"\nData scale: {self.n_obs} observations × {self.n_series} series.")
        if self.n_obs < 500:
            lines.append(
                "Small dataset — prefer simple, regularized models "
                "(Ridge, ElasticNet, low-depth trees). Avoid deep learning."
            )
        elif self.n_obs > 10000:
            lines.append(
                "Large dataset — complex models (LightGBM, XGBoost, "
                "neural networks) are viable. Computational cost matters."
            )

        # ── Cold Start (P1) ─────────────────────────────────
        if self.groups_below_threshold > 0:
            lines.append(
                f"⚠ Cold Start: {self.groups_below_threshold} groups have "
                f"fewer than {self.threshold_sample_size} samples "
                f"(minimum group size={self.min_group_size}). "
                "Predictions for low-sample groups are unreliable. "
                "Consider hierarchical/cross-learning methods."
            )

        return "\n".join(lines)


class FingerprintCalculator:
    """Compute a FeatureFingerprint from a dataset."""

    def calculate(
        self,
        df: pd.DataFrame,
        time_col: Optional[str] = "ds",
        target_col: str = "y",
        id_col: Optional[str] = "unique_id",
    ) -> FeatureFingerprint:
        target = df[target_col].dropna().values
        n_obs = len(target)
        n_series = df[id_col].nunique() if id_col and id_col in df.columns else 1

        zero_ratio = float(np.mean(target == 0))
        std = float(np.std(target)) or 1e-10
        extreme_ratio = float(np.mean(np.abs((target - np.mean(target)) / std) > 3))
        skewness = float(pd.Series(target).skew() or 0)
        kurtosis = float(pd.Series(target).kurtosis() or 0)
        missing_ratio = float(df[target_col].isna().mean())

        acf1 = seasonal_strength = trend_strength = entropy = unitroot_stat = 0.0
        if time_col and time_col in df.columns and len(target) > 2:
            acf1 = self._acf1(target)
            trend_strength = self._trend_strength(target)
            seasonal_strength = self._seasonal_strength(target)
            entropy = self._sample_entropy(target)
            unitroot_stat = self._adf(target)

        seasonality = 7
        freq = "D"
        if time_col and time_col in df.columns:
            dt = pd.to_datetime(df[time_col])
            diffs = dt.diff().dropna()
            if len(diffs) > 0:
                mode = diffs.mode()
                if len(mode) > 0:
                    freq = str(mode.iloc[0])

        return FeatureFingerprint(
            acf1=acf1, seasonal_strength=seasonal_strength,
            trend_strength=trend_strength, entropy=entropy,
            unitroot_stat=unitroot_stat, zero_ratio=zero_ratio,
            extreme_ratio=extreme_ratio, skewness=skewness,
            kurtosis=kurtosis, n_series=n_series, n_obs=n_obs,
            seasonality=seasonality, freq=freq, missing_ratio=missing_ratio,
        )

        # ── Cold Start detection (P1) ──────────────────────────
        if id_col and id_col in df.columns:
            group_sizes = df.groupby(id_col).size()
            fp.min_group_size = int(group_sizes.min()) if len(group_sizes) > 0 else 1
            fp.groups_below_threshold = int((group_sizes < fp.threshold_sample_size).sum())
            if fp.groups_below_threshold > 0:
                fp.tags["is_cold_start"] = True

        return fp

    @staticmethod
    def _acf1(x: np.ndarray) -> float:
        n = len(x)
        if n < 2: return 0.0
        m = np.mean(x)
        d = np.sum((x - m) ** 2)
        return float(np.sum((x[1:] - m) * (x[:-1] - m)) / d) if d > 1e-10 else 0.0

    @staticmethod
    def _trend_strength(x: np.ndarray) -> float:
        n = len(x)
        if n < 3: return 0.0
        from sklearn.linear_model import LinearRegression
        X = np.arange(n).reshape(-1, 1)
        r2 = float(LinearRegression().fit(X, x).score(X, x))
        return max(0.0, min(1.0, r2))

    @staticmethod
    def _seasonal_strength(x: np.ndarray) -> float:
        n = len(x)
        if n < 8: return 0.0
        m = np.mean(x)
        d = np.sum((x - m) ** 2)
        return float(abs(np.sum((x[7:] - m) * (x[:-7] - m))) / d) if d > 1e-10 else 0.0

    @staticmethod
    def _sample_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
        n = len(x)
        if n < m + 2: return 0.0
        std = float(np.std(x)) or 1e-10
        r = r_factor * std

        def _count(t_len: int) -> int:
            templates = np.array([x[i:i + t_len] for i in range(n - t_len + 1)])
            count = 0
            for i in range(len(templates)):
                count += np.sum(np.max(np.abs(templates - templates[i]), axis=1) < r) - 1
            return count
        a = _count(m + 1)
        b = _count(m)
        return float(-np.log(a / b)) if a > 0 and b > 0 else 0.0

    @staticmethod
    def _adf(x: np.ndarray) -> float:
        n = len(x)
        if n < 3: return 0.0
        dy = np.diff(x)
        y1 = x[:-1]
        d = np.sum(y1 ** 2)
        return float(np.sum(y1 * dy) / d) if d > 1e-10 else 0.0