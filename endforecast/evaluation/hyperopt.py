"""
Hyperparameter optimization — grid search, random search, Bayesian optimization.

Supports pluggable backends (Optuna, scikit-learn) and schedules
appropriate for fast/standard/deep experiment tiers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HPOConfig:
    """Hyperparameter optimization configuration.

    Attributes:
        method: "grid", "random", or "bayesian" (optuna).
        n_trials: Number of optimization iterations.
        timeout_seconds: Maximum optimization time.
        cv_folds: Number of CV folds for evaluation.
    """
    method: str = "random"
    n_trials: int = 20
    timeout_seconds: int = 600
    cv_folds: int = 3


@dataclass
class HPOResult:
    """Result of a hyperparameter optimization run.

    Attributes:
        best_params: Optimal hyperparameter dictionary.
        best_score: Best objective value achieved.
        history: List of (params, score) from all trials.
        best_trial: Full trial info from the optimizer.
    """
    best_params: dict[str, Any]
    best_score: float
    history: list[dict] = field(default_factory=list)
    n_trials: int = 0


class HPOptimizer:
    """Pluggable hyperparameter optimization engine.

    Supports three search strategies:
    - grid: Exhaustive search over a parameter grid.
    - random: Random sampling from distributions.
    - bayesian: Optuna-based Bayesian optimization (if installed).

    Usage:
        >>> optimizer = HPOptimizer()
        >>> result = optimizer.optimize(
        ...     objective_fn=train_and_evaluate,
        ...     param_space={"n_estimators": [100, 200, 500], "max_depth": [3, 5, 7]},
        ...     config=HPOConfig(method="random", n_trials=10),
        ... )
        >>> print(result.best_params)
    """

    def optimize(
        self,
        objective_fn: Callable[[dict], float],
        param_space: dict[str, list | tuple],
        config: HPOConfig | None = None,
    ) -> HPOResult:
        """Run hyperparameter optimization.

        Args:
            objective_fn: Function that takes a param dict and returns a score.
                          Lower score is better (minimization).
            param_space: Parameter space specification.
                         For grid/random: {"param": [values]} or
                         {"param": (low, high)} for continuous sampling.
            config: HPO configuration.

        Returns:
            HPOResult with best parameters and optimization history.
        """
        config = config or HPOConfig()

        if config.method == "grid":
            return self._grid_search(objective_fn, param_space, config)
        elif config.method == "random":
            return self._random_search(objective_fn, param_space, config)
        elif config.method == "bayesian":
            return self._bayesian_search(objective_fn, param_space, config)
        else:
            raise ValueError(f"Unknown HPO method: {config.method}")

    def _grid_search(
        self, objective_fn: Callable, param_space: dict, config: HPOConfig,
    ) -> HPOResult:
        """Exhaustive grid search."""
        from itertools import product

        keys = list(param_space.keys())
        values = []
        for k in keys:
            v = param_space[k]
            values.append(v if isinstance(v, list) else list(v))

        history = []
        best_score = float("inf")
        best_params = {}

        for combo in product(*values):
            if len(history) >= config.n_trials:
                break
            params = dict(zip(keys, combo))
            try:
                score = float(objective_fn(params))
                history.append({"params": params, "score": score})
                if score < best_score:
                    best_score = score
                    best_params = params.copy()
            except Exception as exc:
                logger.warning("HPO trial failed: %s", exc)
                history.append({"params": params, "score": float("nan")})

        return HPOResult(
            best_params=best_params,
            best_score=best_score,
            history=history,
            n_trials=len(history),
        )

    def _random_search(
        self, objective_fn: Callable, param_space: dict, config: HPOConfig,
    ) -> HPOResult:
        """Random search over param space."""
        import random
        random.seed(42)

        history = []
        best_score = float("inf")
        best_params = {}

        for _ in range(config.n_trials):
            params = {}
            for k, v in param_space.items():
                if isinstance(v, list):
                    params[k] = random.choice(v)
                elif isinstance(v, tuple) and len(v) == 2:
                    # Continuous range
                    if isinstance(v[0], int):
                        params[k] = random.randint(v[0], v[1])
                    else:
                        params[k] = random.uniform(v[0], v[1])
                else:
                    params[k] = v

            try:
                score = float(objective_fn(params))
                history.append({"params": params, "score": score})
                if score < best_score:
                    best_score = score
                    best_params = params.copy()
            except Exception as exc:
                logger.warning("HPO trial failed: %s", exc)
                history.append({"params": params, "score": float("nan")})

        return HPOResult(
            best_params=best_params,
            best_score=best_score,
            history=history,
            n_trials=len(history),
        )

    def _bayesian_search(
        self, objective_fn: Callable, param_space: dict, config: HPOConfig,
    ) -> HPOResult:
        """Bayesian optimization via Optuna."""
        try:
            import optuna
        except ImportError:
            logger.warning("Optuna not installed. Falling back to random search.")
            return self._random_search(objective_fn, param_space, config)

        best_score = float("inf")
        best_params = {}
        history = []

        def _objective(trial: optuna.Trial) -> float:
            params = {}
            for k, v in param_space.items():
                if isinstance(v, list):
                    if all(isinstance(x, int) for x in v):
                        params[k] = trial.suggest_categorical(k, v)
                    elif all(isinstance(x, float) for x in v):
                        params[k] = trial.suggest_categorical(k, v)
                    else:
                        params[k] = trial.suggest_categorical(k, v)
                elif isinstance(v, tuple) and len(v) == 2:
                    if isinstance(v[0], int):
                        params[k] = trial.suggest_int(k, v[0], v[1])
                    else:
                        params[k] = trial.suggest_float(k, v[0], v[1])
            return objective_fn(params)

        try:
            study = optuna.create_study(direction="minimize")
            study.optimize(
                _objective, n_trials=config.n_trials,
                timeout=config.timeout_seconds,
            )
            best_params = study.best_params
            best_score = float(study.best_value)
            history = [
                {"params": t.params, "score": float(t.value)}
                for t in study.trials if t.value is not None
            ]
        except Exception as exc:
            logger.warning("Optuna optimization failed: %s. Fallback to random.", exc)
            return self._random_search(objective_fn, param_space, config)

        return HPOResult(
            best_params=best_params,
            best_score=best_score,
            history=history,
            n_trials=len(history),
        )