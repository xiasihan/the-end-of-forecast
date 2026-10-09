"""
Controlled experiment router.

Routes each experiment plan to the appropriate execution track
based on data scale, complexity, and user budget constraints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class Route(Enum):
    FAST = "fast"
    STANDARD = "standard"
    DEEP = "deep"


@dataclass
class RouterConfig:
    fast_max_rows: int = 1_000
    standard_max_rows: int = 100_000
    max_trials_fast: int = 10
    max_trials_standard: int = 30
    max_trials_deep: int = 100
    timeout_fast: int = 300
    timeout_standard: int = 1800
    timeout_deep: int = 7200


class ExperimentRouter:
    """Determine the execution track for a given experiment plan."""

    def __init__(self, config: RouterConfig | None = None) -> None:
        self.config = config or RouterConfig()

    def route(self, n_rows: int, n_series: int = 1, n_features: int = 0,
              budget_hint: str | None = None) -> Route:
        if budget_hint and budget_hint != "auto":
            try:
                return Route(budget_hint)
            except ValueError:
                logger.warning("Unknown budget_hint '%s', ignoring.", budget_hint)
        total = n_rows * n_series
        if total <= self.config.fast_max_rows:
            return Route.FAST
        elif total <= self.config.standard_max_rows:
            return Route.STANDARD
        return Route.DEEP

    def get_resource_limits(self, route: Route) -> dict[str, Any]:
        _limits = {
            Route.FAST: {"max_trials": self.config.max_trials_fast, "timeout_seconds": self.config.timeout_fast,
                         "models": ["statistical"], "refinement": "none", "tuning": "none"},
            Route.STANDARD: {"max_trials": self.config.max_trials_standard, "timeout_seconds": self.config.timeout_standard,
                             "models": ["statistical", "ml", "foundation"], "refinement": "statistical", "tuning": "grid"},
            Route.DEEP: {"max_trials": self.config.max_trials_deep, "timeout_seconds": self.config.timeout_deep,
                         "models": ["statistical", "ml", "neural", "foundation"], "refinement": "llm+statistical", "tuning": "bayesian"},
        }
        return _limits.get(route, _limits[Route.STANDARD])