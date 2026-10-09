"""
LeakGuard — Configurable temporal leakage detection framework.

Time leakage is the single most insidious error in predictive modeling.
LeakGuard provides a framework where users define their temporal constraints,
and the engine enforces them strictly at every stage of experimentation.
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class LeakSeverity(Enum):
    PASS = "pass"
    WARNING = "warning"
    BLOCK = "block"


@dataclass
class LeakReport:
    """Structured result of a leakage check run."""

    checks: list[dict] = field(default_factory=list)

    @property
    def all_clear(self) -> bool:
        return not any(c.get("severity") == LeakSeverity.BLOCK.value for c in self.checks)

    @property
    def total_passed(self) -> int:
        return sum(1 for c in self.checks if c.get("severity") == LeakSeverity.PASS.value)

    @property
    def total_warnings(self) -> int:
        return sum(1 for c in self.checks if c.get("severity") == LeakSeverity.WARNING.value)

    @property
    def total_blocked(self) -> int:
        return sum(1 for c in self.checks if c.get("severity") == LeakSeverity.BLOCK.value)

    def summary(self) -> str:
        parts = [f"LeakReport: {self.total_passed}P / {self.total_warnings}W / {self.total_blocked}B"]
        for c in self.checks:
            sev = c.get("severity", "?")
            icon = {"pass": "\u2705", "warning": "\u26a0\ufe0f", "block": "\u26d4"}.get(sev, "?")
            parts.append(f"  {icon} [{c.get('type', '?')}] {c.get('message', '')}")
        return "\n".join(parts)


@dataclass
class LeakRule:
    """A single leakage detection rule defined by the user.

    Supported rule types:
    - "time": Validate training window ends before cutoff.
    - "feature": Validate feature names don't contain forbidden patterns.
    - "label": Validate target column is not included as input feature.
    - "validation": Validate train/validation split has no temporal overlap.
    """

    type: str
    description: str = ""
    cutoff_lag: str = "0d"
    cutoff_time: str = "23:59:00"
    time_column: str = "ds"
    forbidden_patterns: list[str] = field(default_factory=list)
    allowed_lags: list[int] = field(default_factory=list)
    target_column: str = "y"
    isolated_columns: list[str] = field(default_factory=list)
    min_gap: str = "0d"


class LeakGuard:
    """Compiles user-defined leak rules into an executable check chain."""

    def __init__(self, rules: list[LeakRule]) -> None:
        self.rules = rules

    @classmethod
    def from_config(cls, config: dict) -> "LeakGuard":
        rules: list[LeakRule] = []
        time_cfg = config.get("time_leak_protection", {})
        if time_cfg.get("enabled", True):
            rules.append(LeakRule(
                type="time",
                cutoff_lag=time_cfg.get("cutoff_lag", "3d"),
                cutoff_time=time_cfg.get("cutoff_time", "23:59:00"),
                time_column=config.get("time_column", "ds"),
                description=time_cfg.get("description", "Time boundary protection"),
            ))
        fc_cfg = config.get("feature_crossing", {})
        if fc_cfg.get("enabled", True):
            rules.append(LeakRule(
                type="feature",
                forbidden_patterns=fc_cfg.get("forbidden_patterns", []),
                allowed_lags=fc_cfg.get("allowed_lags", []),
                description=fc_cfg.get("description", "Feature crossing detection"),
            ))
        ll_cfg = config.get("label_leakage", {})
        if ll_cfg.get("enabled", True):
            rules.append(LeakRule(
                type="label",
                target_column=ll_cfg.get("target_column", "y"),
                isolated_columns=ll_cfg.get("isolated_columns", ["y"]),
                description=ll_cfg.get("description", "Label leakage detection"),
            ))
        vg_cfg = config.get("validation_gap", {})
        if vg_cfg.get("enabled", True):
            rules.append(LeakRule(
                type="validation",
                min_gap=vg_cfg.get("min_gap", "0d"),
                time_column=config.get("time_column", "ds"),
                description=vg_cfg.get("description", "Validation gap enforcement"),
            ))
        return cls(rules)

    def check(
        self,
        prediction_date: str | pd.Timestamp | None,
        training_df: pd.DataFrame,
        feature_columns: Optional[list[str]] = None,
        val_start: Optional[str | pd.Timestamp] = None,
    ) -> LeakReport:
        report = LeakReport()
        feature_columns = feature_columns or []
        for rule in self.rules:
            if rule.type == "time":
                res = self._check_time(rule, prediction_date, training_df)
            elif rule.type == "feature":
                res = self._check_features(rule, feature_columns)
            elif rule.type == "label":
                res = self._check_label(rule, feature_columns)
            elif rule.type == "validation":
                res = self._check_validation(rule, training_df, val_start)
            else:
                res = {"type": rule.type, "severity": LeakSeverity.PASS.value, "message": "custom"}
            report.checks.append(res)
        return report

    def _check_time(self, rule: LeakRule, prediction_date, df: pd.DataFrame) -> dict:
        if prediction_date is None:
            return {"type": "time", "severity": LeakSeverity.PASS.value, "message": "No prediction date; skip."}
        cutoff = self._parse_cutoff(prediction_date, rule)
        if rule.time_column not in df.columns:
            return {"type": "time", "severity": LeakSeverity.WARNING.value, "message": f"Time column '{rule.time_column}' missing."}
        max_date = pd.Timestamp(df[rule.time_column].max())
        if max_date > cutoff:
            return {"type": "time", "severity": LeakSeverity.BLOCK.value, "message": f"Data exceeds cutoff. max={max_date} cutoff={cutoff}"}
        return {"type": "time", "severity": LeakSeverity.PASS.value, "message": f"Cutoff respected: {cutoff}"}

    def _check_features(self, rule: LeakRule, columns: list[str]) -> dict:
        violations = [c for c in columns for p in rule.forbidden_patterns if fnmatch.fnmatch(c, p)]
        if violations:
            return {"type": "feature", "severity": LeakSeverity.BLOCK.value, "message": f"Forbidden patterns: {violations}"}
        return {"type": "feature", "severity": LeakSeverity.PASS.value, "message": "Clean"}

    def _check_label(self, rule: LeakRule, columns: list[str]) -> dict:
        overlap = set(columns) & set(rule.isolated_columns or [])
        if overlap:
            return {"type": "label", "severity": LeakSeverity.BLOCK.value, "message": f"Label leakage: {overlap}"}
        return {"type": "label", "severity": LeakSeverity.PASS.value, "message": "Clean"}

    def _check_validation(self, rule: LeakRule, train_df: pd.DataFrame, val_start) -> dict:
        if val_start is None:
            return {"type": "validation", "severity": LeakSeverity.PASS.value, "message": "No validation set."}
        if rule.time_column not in train_df.columns:
            return {"type": "validation", "severity": LeakSeverity.WARNING.value, "message": "No time column."}
        train_end = pd.Timestamp(train_df[rule.time_column].max())
        val_begin = pd.Timestamp(val_start)
        gap = pd.Timedelta(rule.min_gap)
        if val_begin < train_end + gap:
            return {"type": "validation", "severity": LeakSeverity.BLOCK.value, "message": f"Overlap: train_end={train_end} val_start={val_begin}"}
        return {"type": "validation", "severity": LeakSeverity.PASS.value, "message": "Clean"}

    @staticmethod
    def _parse_cutoff(prediction_date, rule: LeakRule) -> pd.Timestamp:
        pred = pd.Timestamp(str(prediction_date))
        lag = pd.Timedelta(rule.cutoff_lag)
        cutoff = pred - lag
        if rule.cutoff_time:
            h, m, s = (int(x) for x in rule.cutoff_time.split(":"))
            cutoff = cutoff.replace(hour=h, minute=m, second=s)
        return cutoff