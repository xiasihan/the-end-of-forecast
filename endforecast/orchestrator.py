"""
Main orchestrator — complete 10-phase prediction pipeline per
the standard predictive modeling methodology.

Phases:
  0. Requirements Gathering
  1. Data Understanding (Exploration + EDA)
  2. Data Preparation (Preprocessing)
  3. LeakGuard Check
  4. Baseline Establishment
  5. Experiment Planning & Execution
  6. Evaluation & Error Analysis
  7. Iterative Refinement (Diagnostician + Refiner)
  8. Interpretation & Explainability
  9. Deployment
 10. Drift Detection

All phases are task-agnostic (classification / regression / time series).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from endforecast.config import EndForecastConfig
from endforecast.agents.requirements import RequirementsGatherer, RequirementsSpec
from endforecast.agents.explorer import DataExplorer, ExplorationReport
from endforecast.agents.planner import ExperimentPlanner, ExperimentPlan
from endforecast.agents.diagnostician import Diagnostician, DiagnosisReport
from endforecast.agents.refiner import Refiner, RefinementResult
from endforecast.engine.pipeline import Pipeline, PipelineConfig, PipelineResult
from endforecast.engine.leakguard import LeakGuard, LeakReport
from endforecast.engine.router import ExperimentRouter, Route
from endforecast.engine.trial import Trial, TrialResult, TrialState, OptimizationBudget
from endforecast.engine.baseline import BaselineRunner, BaselineResult
from endforecast.engine.explainability import Explainer, ExplainabilityResult
from endforecast.engine.drift import DriftDetector, DriftReport
from endforecast.engine.fingerprint import FingerprintCalculator
from endforecast.evaluation.splitter import DataSplitter, SplitConfig, NestedCVSplitter, NestedCVConfig
from endforecast.evaluation.metrics import MetricCalculator, MetricResult, StatisticalComparison, StabilityEvaluator, CalibrationMetrics
from endforecast.evaluation.hyperopt import HPOptimizer, HPOConfig
from endforecast.hooks import PipelineHooks

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Complete result from an EndForecast run across all 10 phases."""

    # Core artifacts
    pipeline: Optional[Pipeline] = None
    requirements: Optional[RequirementsSpec] = None
    report: Optional[ExplorationReport] = None
    baseline_result: Optional[BaselineResult] = None
    plan: Optional[ExperimentPlan] = None
    best_trial: Optional[TrialResult] = None
    explainability_result: Optional[ExplainabilityResult] = None
    leak_report: Optional[LeakReport] = None
    drift_report: Optional[DriftReport] = None

    # History & governance
    history: list[TrialResult] = field(default_factory=list)
    optimization_budget: Optional[OptimizationBudget] = None
    errors: list[str] = field(default_factory=list)

    # Data snooping tax — three tiers of metric credibility
    metric_tiers: dict = field(default_factory=lambda: {
        "training_score": float("nan"),
        "validation_score": float("nan"),
        "holdout_score": float("nan"),
        "optimization_rounds": 0,
        "data_snooping_warning": False,
    })
    # Stability estimate
    metric_stability: Optional[Any] = None  # StabilityResult

    def export(self, mode: str = "code", output_dir: str | Path = "./export") -> Path:
        if self.pipeline is None:
            raise RuntimeError("No pipeline to export. Run was not successful.")
        return self.pipeline.export(mode=mode, output_dir=output_dir)

    @property
    def success(self) -> bool:
        return self.pipeline is not None and len(self.errors) == 0

    def summary(self) -> str:
        """Multi-line summary of the entire run."""
        lines = ["=" * 60, "EndForecast Run Summary", "=" * 60]
        if self.requirements:
            lines.append(f"  Goal: {self.requirements.business_purpose[:80]}")
        if self.report:
            lines.append(f"  Task Type: {self.report.task_type}")
        if self.baseline_result and self.baseline_result.best:
            lines.append(f"  Baseline: {self.baseline_result.best.trial.model}={self.baseline_result.best.metric_value:.4f}")
        if self.best_trial:
            lines.append(f"  Best Model: {self.best_trial.trial.model} ({self.best_trial.trial.metric}={self.best_trial.metric_value:.4f})")
            if self.baseline_result and self.baseline_result.best:
                impr = (self.best_trial.metric_value - self.baseline_result.baseline_floor) / self.baseline_result.baseline_floor * 100
                lines.append(f"  Improvement vs Baseline: {impr:.1f}%")
        if self.metric_tiers.get("refinement_score") is not None:
            lines.append(f"  Refinement Score: {self.metric_tiers['refinement_score']:.4f}")
        if self.metric_tiers.get("ensemble_score") is not None:
            lines.append(f"  Ensemble Score: {self.metric_tiers['ensemble_score']:.4f}")
        if self.explainability_result and hasattr(self.explainability_result, 'top_features'):
            top_feats = self.explainability_result.top_features
            if hasattr(top_feats, '__iter__'):
                lines.append(f"  Top Features: {list(top_feats)[:3]}")
        if self.optimization_budget:
            lines.append(f"  Degrees of Freedom: {self.optimization_budget.summary()}")
        if self.metric_tiers:
            lines.append(f"  Holdout Score: {self.metric_tiers.get('holdout_score', float('nan'))}")
            if self.metric_tiers.get("data_snooping_warning"):
                lines.append("  ⚠ DATA SNOOPING WARNING: validation_score may be optimistic")
            if self.metric_tiers.get("heuristic_first_warning"):
                lines.append(f"  ⚠ HEURISTIC SUFFICIENT: {self.metric_tiers.get('heuristic_first_message', '')}")
            if self.metric_tiers.get("training_data_hash"):
                lines.append(f"  Model Lineage: hash={self.metric_tiers.get('training_data_hash')} | {self.metric_tiers.get('n_training_rows')} rows")
            if self.metric_tiers.get("feature_dependencies"):
                deps = self.metric_tiers["feature_dependencies"]
                lines.append(f"  Feature Dependencies: {len(deps.get('must_be_available', []))} features required at serving time")
        if self.errors:
            lines.append(f"  Errors: {len(self.errors)}")
        lines.append("=" * 60)
        return "\n".join(lines)


class EndForecast:
    """Complete 10-phase auto-prediction pipeline.

    Usage:
        >>> ef = EndForecast()
        >>> result = ef.run("sales.csv", target_col="revenue",
        ...                 requirements="Predict next month's revenue")
        >>> print(result.summary())
        >>> result.export("code", output_dir="./my_app")
    """

    def __init__(self, config: Optional[EndForecastConfig] = None) -> None:
        self.config = config or EndForecastConfig()
        self.gatherer = RequirementsGatherer()
        self.explorer = DataExplorer()
        self.router = ExperimentRouter()
        self.planner = ExperimentPlanner()
        self.diagnostician = Diagnostician()
        self.refiner = Refiner()
        self.splitter = DataSplitter()
        self.metrics_calc = MetricCalculator()
        self.hyperopt = HPOptimizer()
        self.baseline_runner = BaselineRunner(metrics_calculator=self.metrics_calc)
        self.explainer = Explainer()
        self.drift_detector = DriftDetector()
        self.ncv_splitter = NestedCVSplitter()
        self.stat_compare = StatisticalComparison()
        self.stability_eval = StabilityEvaluator(metric_calculator=self.metrics_calc)
        self.calibration = CalibrationMetrics()

    def run(
        self,
        data: str | Path | pd.DataFrame,
        time_col: str = "ds",
        target_col: str = "y",
        id_col: Optional[str] = "unique_id",
        leak_rules: Optional[dict[str, Any]] = None,
        requirements: Optional[str] = None,
        budget: str = "auto",
        hooks: Optional[PipelineHooks] = None,
    ) -> RunResult:
        result = RunResult()

        if hooks is None:
            hooks = PipelineHooks()  # no-op hooks, never null

        # ── Initialize optimization budget ────────────────────────
        opt_budget = OptimizationBudget(
            max_decisions=self.config.experiment.max_rounds * 3,
        )
        result.optimization_budget = opt_budget
        logger.info("Optimization budget: %s decisions available", opt_budget.max_decisions)

        # ═══ Phase 0: Requirements ═══════════════════════════════
        _t0 = time.time()
        hooks.fire_phase_start(0, "Requirements Gathering")
        logger.info("=" * 50)
        logger.info("Phase 0: Requirements Gathering")
        if requirements:
            result.requirements = self.gatherer.gather(description=requirements)
            logger.info("  Task hint: %s", result.requirements.task_type_hint)
        hooks.fire_phase_complete(0, "Requirements Gathering", "completed",
                                   data={"task_type_hint": result.requirements.task_type_hint if result.requirements else None},
                                   elapsed_ms=int((time.time() - _t0) * 1000))

        # ═══ Phase 1: Data Understanding ══════════════════════════
        _t0 = time.time()
        hooks.fire_phase_start(1, "Data Understanding")
        logger.info("Phase 1: Data Understanding")
        df = self._load_data(data)
        logger.info("  Loaded %s rows × %s cols", len(df), len(df.columns))

        report = self.explorer.explore(df, time_col=time_col, target_col=target_col, id_col=id_col)
        result.report = report
        logger.info("  Diagnostic narrative:\n%s", report.fingerprint.diagnostic)
        if report.fingerprint.tags.get("is_cold_start"):
            result.metric_tiers["cold_start_detected"] = True
            result.metric_tiers["cold_start_groups"] = report.fingerprint.groups_below_threshold
        logger.info("  Task: %s | %s", report.task_type, report.summary())
        hooks.fire_phase_complete(1, "Data Understanding", "completed",
                                   data={"task_type": report.task_type, "n_rows": report.n_rows,
                                         "n_cols": report.n_cols, "missing_ratio": report.missing_ratio,
                                         "quality_flags": report.quality_flags,
                                         "diagnostic": report.fingerprint.diagnostic[:500]},
                                   elapsed_ms=int((time.time() - _t0) * 1000))

        # ═══ Phase 2: Data Preparation ════════════════════════════
        _t0 = time.time()
        hooks.fire_phase_start(2, "Data Preparation")
        logger.info("Phase 2: Data Preparation")
        preproc = self._plan_preprocessing(report)
        logger.info("  Steps: %s", preproc)
        hooks.fire_phase_complete(2, "Data Preparation", "completed",
                                   data={"steps": preproc},
                                   elapsed_ms=int((time.time() - _t0) * 1000))

        # ═══ Phase 3: LeakGuard ══════════════════════════════════
        _t0 = time.time()
        if leak_rules:
            hooks.fire_phase_start(3, "LeakGuard Check")
            logger.info("Phase 3: LeakGuard Check")
            guard = LeakGuard.from_config(leak_rules)
            lr = guard.check("check", df)
            result.leak_report = lr
            status = "completed" if lr.all_clear else "failed"
            hooks.fire_phase_complete(3, "LeakGuard Check", status,
                                       data={"all_clear": lr.all_clear, "summary": lr.summary()},
                                       elapsed_ms=int((time.time() - _t0) * 1000))
            if not lr.all_clear:
                result.errors.append(f"LeakGuard BLOCKED: {lr.summary()}")
                return result
            logger.info("  PASSED")
        else:
            hooks.fire_phase_complete(3, "LeakGuard Check", "completed",
                                       data={"skipped": True},
                                       elapsed_ms=0)

        # ═══════════════════════════════════════════════════════════
        # ── LLM-driven experiment configuration ────────────────
        #   One LLM call decides: holdout, cv, route, tuning, metric.
        #   Must run BEFORE Phase 4 so baseline uses the same metric.
        #   Replaces ~8 hardcoded if/elif branches.
        # ═══════════════════════════════════════════════════════════
        n_raw_features = len(df.columns) - len({target_col, time_col, id_col, "date_time", "unique_id"} & set(df.columns))
        n_series_val = df[id_col].nunique() if (id_col and id_col in df.columns) else 1

        llm_exp_config: Optional[dict] = None
        try:
            from endforecast._llm import experiment_configurator
            llm_exp_config = experiment_configurator(
                report.fingerprint.diagnostic[:4000],
                report.task_type, len(df), n_series_val, n_raw_features,
            )
            if llm_exp_config:
                logger.info("  LLM experiment config: holdout=%s, cv=%s, route=%s, tuning=%s, metric=%s",
                            llm_exp_config.get("holdout", {}).get("method", "?"),
                            llm_exp_config.get("cv", {}).get("method", "?"),
                            llm_exp_config.get("route", {}).get("name", "?"),
                            llm_exp_config.get("tuning", {}).get("method", "?"),
                            llm_exp_config.get("primary_metric", "?"))
                logger.info("  rationale: %s", llm_exp_config.get("rationale", "")[:100])
                hooks.fire_llm_decision("configurator",
                    decision_summary=f"holdout={llm_exp_config.get('holdout',{}).get('method','?')}, "
                                    f"cv={llm_exp_config.get('cv',{}).get('method','?')}, "
                                    f"route={llm_exp_config.get('route',{}).get('name','?')}, "
                                    f"metric={llm_exp_config.get('primary_metric','?')}",
                    rationale=llm_exp_config.get("rationale", ""))
        except Exception as exc:
            logger.warning("  LLM experiment configurator unavailable: %s", exc)

        # ── 1. Primary metric (LLM or fallback) ────────────────
        primary_metric: Optional[str] = None
        if llm_exp_config:
            llm_metric = llm_exp_config.get("primary_metric")
            if llm_metric and isinstance(llm_metric, str):
                available = self.metrics_calc.list_metrics(report.task_type)
                if llm_metric in available:
                    primary_metric = llm_metric
        if primary_metric is None:
            primary_metric = report.suggested_metrics[0] if report.suggested_metrics else None

        # ── 2. Holdout split (LLM or fallback) ──────────────────
        holdout_cfg = llm_exp_config.get("holdout", {}) if llm_exp_config else {}
        holdout_method = holdout_cfg.get("method", "chronological" if report.task_type == "timeseries" else ("stratified" if report.task_type == "classification" else "random"))
        holdout_ratio = holdout_cfg.get("ratio", self.config.experiment.holdout_ratio)

        df_holdout = None
        _holdout_size = max(1, int(len(df) * holdout_ratio))
        if len(df) < 20:
            _holdout_size = 0

        if _holdout_size > 0:
            if holdout_method == "chronological" and time_col and time_col in df.columns:
                df = df.sort_values(time_col).reset_index(drop=True)
                df_holdout = df.iloc[-_holdout_size:].copy()
                df = df.iloc[:-_holdout_size].copy()
                logger.info("  Holdout: last %d rows (%.1f%%), chronological [LLM=%s]",
                            len(df_holdout), holdout_ratio * 100, bool(llm_exp_config))
            elif holdout_method == "stratified":
                from sklearn.model_selection import train_test_split as tts
                y_all = df[target_col].values
                idx_all = np.arange(len(df))
                train_idx, holdout_idx = tts(
                    idx_all, test_size=holdout_ratio, stratify=y_all,
                    random_state=42,
                )
                df_holdout = df.iloc[holdout_idx].copy()
                df = df.iloc[train_idx].copy()
                logger.info("  Holdout: %d rows (%.1f%%), stratified [LLM=%s]",
                            len(df_holdout), holdout_ratio * 100, bool(llm_exp_config))
            else:  # "random" or unknown
                idx_all = np.arange(len(df))
                rng = np.random.RandomState(42)
                holdout_idx = rng.choice(idx_all, size=_holdout_size, replace=False)
                df_holdout = df.iloc[holdout_idx].copy()
                df = df.drop(df.index[holdout_idx]).copy()
                logger.info("  Holdout: %d rows (%.1f%%), random [LLM=%s]",
                            len(df_holdout), holdout_ratio * 100, bool(llm_exp_config))
        logger.info("  Training/eval on %d rows after holdout", len(df))

        # ── 3. Route & trial budget (LLM or fallback) ──────────
        route_cfg = llm_exp_config.get("route", {}) if llm_exp_config else {}
        llm_route_name = route_cfg.get("name")
        llm_max_trials = route_cfg.get("max_trials")

        if llm_route_name:
            try:
                route = Route(llm_route_name)
                limits = self.router.get_resource_limits(route)
                if llm_max_trials and isinstance(llm_max_trials, int):
                    limits["max_trials"] = llm_max_trials
            except ValueError:
                route = self.router.route(n_rows=len(df), n_series=n_series_val, budget_hint=budget)
                limits = self.router.get_resource_limits(route)
        else:
            route = self.router.route(n_rows=len(df), n_series=n_series_val, budget_hint=budget)
            limits = self.router.get_resource_limits(route)

        # ── 4. Tuning method (LLM or fallback) ─────────────────
        tuning_cfg = llm_exp_config.get("tuning", {}) if llm_exp_config else {}
        llm_tuning = tuning_cfg.get("method")
        round1_tuning = "none"  # Round 1 never tunes
        later_tuning = llm_tuning if llm_tuning and llm_tuning != "none" else limits.get("tuning", "none")

        # ═══ Phase 4: Baseline Establishment ══════════════════════
        _t0 = time.time()
        hooks.fire_phase_start(4, "Baseline Establishment")
        logger.info("Phase 4: Baseline Establishment")
        result.baseline_result = self.baseline_runner.run(
            df, target_col=target_col, time_col=time_col, id_col=id_col,
            task_type=report.task_type, metric=primary_metric or "auto",
        )
        baseline_floor = result.baseline_result.baseline_floor
        logger.info("  Baseline floor: %.4f (%s)", baseline_floor, primary_metric)
        hooks.fire_phase_complete(4, "Baseline Establishment", "completed",
                                   data={"baseline_floor": baseline_floor,
                                         "best_baseline": result.baseline_result.best.trial.model if result.baseline_result.best else "?",
                                         "metric": primary_metric,
                                         "baselines": [{"model": r.trial.model, "value": r.metric_value}
                                                       for r in result.baseline_result.results]},
                                   elapsed_ms=int((time.time() - _t0) * 1000))

        # ── Heuristic First check ───────────────────────────────
        if requirements and result.requirements and result.requirements.acceptable_min_performance:
            if abs(baseline_floor) <= result.requirements.acceptable_min_performance:
                logger.info(
                    "  ⚠ Heuristic First: baseline (%.4f) already meets "
                    "acceptable performance (%.4f). ML may not be necessary.",
                    baseline_floor, result.requirements.acceptable_min_performance,
                )
                result.metric_tiers["heuristic_first_warning"] = True
                result.metric_tiers["heuristic_first_message"] = (
                    f"Baseline already achieves target performance "
                    f"({baseline_floor:.4f} ≤ {result.requirements.acceptable_min_performance}). "
                    f"Consider deploying the baseline directly instead of ML."
                )
            else:
                result.metric_tiers["heuristic_first_warning"] = False

        # ═══ Phase 5: Experiment Planning & Execution ═════════════
        logger.info("Phase 5: Experiment Planning & Execution")

        # ── LLM-driven experiment planning ────────────────────────
        llm_plan_suggestions = None
        try:
            from endforecast._llm import planner_plan
            diag = report.fingerprint.diagnostic[:4000]
            llm_plan_suggestions = planner_plan(diag, report.task_type, len(df))
            if llm_plan_suggestions:
                logger.info("  LLM planner: %s models, %s feature combos",
                            len(llm_plan_suggestions.get("models", [])),
                            len(llm_plan_suggestions.get("features", [])))
                hooks.fire_llm_decision("planner",
                    decision_summary=f"{len(llm_plan_suggestions.get('models',[]))} models, "
                                    f"{len(llm_plan_suggestions.get('features',[]))} feature combos",
                    rationale=llm_plan_suggestions.get("rationale", ""))
        except Exception as exc:
            logger.warning("  LLM planner unavailable, using fallback: %s", exc)

        plan = self.planner.plan(report, route=route, max_trials=limits["max_trials"],
                                 llm_suggestions=llm_plan_suggestions,
                                 primary_metric=primary_metric)
        result.plan = plan
        logger.info("  Route: %s | Trials: %s | %s", route.value, plan.n_trials,
                    "LLM-driven" if plan.llm_driven else "fallback")

        history: list[TrialResult] = []
        best: Optional[TrialResult] = None
        prev_best: Optional[float] = None

        # ── Store best predictions & truth for refinement/ensemble ──
        best_predictions: Optional[np.ndarray] = None
        best_truth: Optional[np.ndarray] = None
        best_val_indices: Optional[np.ndarray] = None  # row indices into df

        for round_num in range(1, self.config.experiment.max_rounds + 1):
            if opt_budget.is_exhausted:
                logger.warning("  Optimization budget exhausted at round %s", round_num - 1)
                break

            _rt0 = time.time()
            hooks.fire_round_start(round_num, plan.n_trials)
            logger.info("  --- Round %s/%s ---", round_num, self.config.experiment.max_rounds)
            round_results = self._execute_round(plan, df, target_col, time_col, round_num, preproc, later_tuning if round_num > 1 else round1_tuning)
            if not round_results:
                logger.warning("    No successful trials.")
                continue
            opt_budget.consume(f"Round {round_num}: executed {len(round_results)} trials")
            history.extend(round_results)
            # ── Direction-aware best selection ─────────────────
            # "minimize": lower is better (mae, mase, rmse)
            # "maximize": higher is better (f1, accuracy, auc, r2)
            metric_dir = self.metrics_calc._DIRECTION.get(
                best.trial.metric if best else (primary_metric or report.suggested_metrics[0]),
                "minimize",
            )
            if metric_dir == "maximize":
                round_best = max(round_results, key=lambda r: r.metric_value)
                if best is None:
                    is_better = True
                else:
                    is_better = round_best.metric_value > best.metric_value
            else:
                round_best = min(round_results, key=lambda r: r.metric_value)
                if best is None:
                    is_better = True
                else:
                    is_better = round_best.metric_value < best.metric_value

            if best is None or is_better:
                best = round_best
                opt_budget.consume(f"Round {round_num}: new best model selected ({round_best.trial.model})")
            logger.info("    Best: %s %s=%.4f±%.4f (n=%s)", best.trial.pipeline_id, best.trial.metric.upper(), best.metric_value, best.metric_std, best.metric_n_runs)

            # Check baseline requirement (computed after best is updated)
            if metric_dir == "maximize":
                _beats = best.metric_value > baseline_floor
            else:
                _beats = best.metric_value < baseline_floor
            if not _beats and baseline_floor != float("inf"):
                logger.info("    ⚠ Warning: Best model NOT better than baseline (%.4f)", baseline_floor)

            # ── Direction-aware convergence check ─────────────
            llm_diag_output = None  # diagnosis captured earlier if LLM available
            if prev_best is not None and best is not None and prev_best != 0:
                if metric_dir == "maximize":
                    improvement = (best.metric_value - prev_best) / abs(prev_best) if prev_best > 0 else 0
                else:
                    improvement = abs((best.metric_value - prev_best) / prev_best)
                if improvement < self.config.experiment.convergence_threshold:
                    logger.info("    Converged (impr %.2f%% < %.1f%%)", improvement * 100, self.config.experiment.convergence_threshold * 100)
                    opt_budget.consume(f"Round {round_num}: convergence declared")
                    hooks.fire_round_complete(round_num, plan.n_trials,
                        best_model=best.trial.model, best_metric=best.metric_value,
                        best_metric_name=best.trial.metric,
                        trial_summaries=[{"id": r.trial.id, "model": r.trial.model,
                                          "value": r.metric_value, "std": r.metric_std}
                                         for r in round_results],
                        diagnosis=None,
                        elapsed_ms=int((time.time() - _rt0) * 1000))
                    break
            prev_best = best.metric_value

        result.history = history
        result.best_trial = best
        logger.info("  Trials completed: %s | Best: %.4f", len(history), best.metric_value if best else float("nan"))

        # ─── Retrain best model on full df to get predictions for holdout ───
        if best and best.is_successful:
            try:
                actual_preproc = best.trial.preprocess if best.trial.preprocess else preproc
                best_cfg = PipelineConfig(
                    pipeline_id="best_full", task_type=report.task_type,
                    preprocess=actual_preproc, features=best.trial.features,
                    model={"type": best.trial.model, "params": best.trial.model_params},
                    cv_strategy=best.trial.cv_strategy,
                )
                best_pipe = Pipeline(best_cfg)
                best_pipe.fit(df, target_col=target_col, time_col=time_col)
                best_predictions = best_pipe.predict(df, target_col=target_col, time_col=time_col).predictions
                if df_holdout is not None:
                    holdout_preds = best_pipe.predict(df_holdout, target_col=target_col, time_col=time_col).predictions
                    best_truth = df_holdout[target_col].values
                    if len(holdout_preds) < len(best_truth):
                        best_truth = best_truth[-len(holdout_preds):]
                    best_predictions = holdout_preds
                logger.debug("  Best model retrained on full data, saved predictions for refinement")
            except Exception as exc:
                logger.warning("  Best model retrain for refinement failed: %s", exc)

        # ═══ Phase 6: Evaluation & Error Analysis ═══════════════
        _t0 = time.time()
        hooks.fire_phase_start(6, "Evaluation & Error Analysis")
        logger.info("Phase 6: Evaluation & Error Analysis")

        # ── Holdout evaluation (single unbiased assessment) ─────
        holdout_score = float("nan")
        if best_predictions is not None and best_truth is not None and df_holdout is not None:
            try:
                ho_mr = self.metrics_calc.evaluate(
                    best_truth, best_predictions,
                    task_type=report.task_type,
                    metrics=[best.trial.metric],
                )
                holdout_score = ho_mr.value
                logger.info("  Holdout score: %.4f (unbiased estimate on %d held-out rows)",
                            holdout_score, len(df_holdout))
            except Exception as exc:
                logger.warning("  Holdout evaluation failed: %s", exc)

        result.metric_tiers = {
            "training_score": float("nan"),
            "validation_score": best.metric_value if best else float("nan"),
            "holdout_score": holdout_score,
            "optimization_rounds": len(history),
            "data_snooping_warning": False,
        }
        hooks.fire_phase_complete(6, "Evaluation & Error Analysis", "completed",
                                   data={"holdout_score": holdout_score,
                                         "validation_score": best.metric_value if best else None},
                                   elapsed_ms=int((time.time() - _t0) * 1000))

        if best and best.is_successful and best.evaluation_df is not None:
            y_true = best.evaluation_df[target_col].values
            y_pred = best.metrics_detail.get("predictions", None)
            if y_pred is not None:
                mr = self.metrics_calc.evaluate(y_true, y_pred, task_type=report.task_type)
                logger.info("  %s", mr.interpretation)

        # ═══ Phase 7: Refinement ═════════════════════════════════
        _t0 = time.time()
        hooks.fire_phase_start(7, "Iterative Refinement")
        refinement_params = {}
        if best_predictions is not None and best_truth is not None and best and best.is_successful:
            logger.info("Phase 7: Iterative Refinement")
            try:
                # ── LLM-driven refinement decision ────────────────
                ref_method = "statistical"
                ref_params: dict[str, Any] = {}
                try:
                    from endforecast._llm import refiner_decide
                    # Provide residual statistics so LLM can decide
                    residuals = best_truth - best_predictions
                    error_brief = json.dumps({
                        "task_type": report.task_type,
                        "best_metric": best.metric_value,
                        "best_metric_name": best.trial.metric,
                        "baseline": baseline_floor,
                        "residual_mean": round(float(np.mean(residuals)), 4),
                        "residual_std": round(float(np.std(residuals)), 4),
                        "residual_p5": round(float(np.percentile(residuals, 5)), 4),
                        "residual_p95": round(float(np.percentile(residuals, 95)), 4),
                        "metric_std": best.metric_std,
                    }, ensure_ascii=False)
                    llm_ref = refiner_decide(error_brief, report.task_type)
                    if llm_ref and llm_ref.get("apply"):
                        ref_method = llm_ref.get("method", "statistical")
                        ref_params = llm_ref.get("params", {})
                        logger.info("  LLM refiner: %s (rationale: %s)", ref_method,
                                    llm_ref.get("rationale", "")[:60])
                except Exception as exc:
                    logger.debug("  LLM refiner unavailable, using default: %s", exc)

                # ── Apply refinement to actual predictions ────────
                ref_result = self.refiner.refine(
                    best_predictions, report.task_type, method=ref_method,
                    reference=best_truth, **ref_params,
                )
                logger.info("  Method: %s | Improvement: %.1f%%", ref_result.method, ref_result.improvement_pct)
                refinement_params = ref_result.details

                # ── Recompute metric on refined predictions ───────
                try:
                    ref_mr = self.metrics_calc.evaluate(
                        best_truth, ref_result.refined_predictions,
                        task_type=report.task_type, metrics=[best.trial.metric],
                    )
                    result.metric_tiers["refinement_score"] = ref_mr.value
                    logger.info("  Refined score: %.4f (was %.4f)", ref_mr.value, holdout_score if not np.isnan(holdout_score) else best.metric_value)
                except Exception:
                    pass
            except Exception as exc:
                logger.warning("  Skipped: %s", exc)

        # ── LLM-driven ensemble decision ─────────────────────────
        if len(history) >= 4 and best and best.is_successful:
            logger.info("  Ensemble Analysis")
            try:
                from endforecast._llm import ensemble_decide
                trial_summary = json.dumps([
                    {"id": r.trial.id, "model": r.trial.model, "value": r.metric_value,
                     "std": r.metric_std, "features": r.trial.features}
                    for r in sorted(history, key=lambda x: x.metric_value)[:8]
                ], ensure_ascii=False)
                ensemble_plan = ensemble_decide(trial_summary)
                if ensemble_plan and ensemble_plan.get("mix"):
                    logger.info("  Ensemble: %s → %s (rationale: %s)",
                                ensemble_plan.get("method", "?"),
                                ensemble_plan.get("candidate_models", []),
                                ensemble_plan.get("rationale", "")[:60])
                    result.metric_tiers["ensemble_strategy"] = ensemble_plan

                    # ── Execute ensemble: train top-k models, combine predictions ──
                    candidate_ids = ensemble_plan.get("candidate_models", [])
                    if candidate_ids and best_predictions is not None and best_truth is not None:
                        try:
                            ensemble_preds = self._execute_ensemble(
                                history, candidate_ids, df, df_holdout,
                                target_col, time_col, preproc,
                                ensemble_plan.get("method", "median_ensemble"),
                            )
                            if ensemble_preds is not None:
                                ens_mr = self.metrics_calc.evaluate(
                                    best_truth, ensemble_preds,
                                    task_type=report.task_type, metrics=[best.trial.metric],
                                )
                                result.metric_tiers["ensemble_score"] = ens_mr.value
                                logger.info("  Ensemble score: %.4f (best single: %.4f)",
                                            ens_mr.value,
                                            holdout_score if not np.isnan(holdout_score) else best.metric_value)
                        except Exception as exc:
                            logger.warning("  Ensemble execution failed: %s", exc)
            except Exception as exc:
                logger.debug("  Ensemble decision skipped: %s", exc)

        # ═══ Phase 8: Interpretation ═════════════════════════════
        logger.info("Phase 8: Interpretation & Explainability")
        if best and best.is_successful:
            try:
                actual_preproc = best.trial.preprocess if best.trial.preprocess else preproc
                cfg_temp = PipelineConfig(
                    pipeline_id="temp_explain", task_type=report.task_type,
                    preprocess=actual_preproc,
                    model={"type": best.trial.model, "params": best.trial.model_params},
                    features=best.trial.features,
                )
                pipe_temp = Pipeline(cfg_temp)
                pipe_temp.fit(df, target_col=target_col, time_col=time_col)
                X_explain = df.drop(columns=[target_col], errors="ignore").select_dtypes(include=[np.number])
                if len(X_explain.columns) > 0:
                    result.explainability_result = self.explainer.explain(pipe_temp._model, X_explain.fillna(0))
                    logger.info("  %s", result.explainability_result.summary())
            except Exception as exc:
                logger.warning("  Explainability skipped: %s", exc)

        # ═══ Phase 9: Pipeline Export + Model Lineage ════════════
        logger.info("Phase 9: Pipeline Construction & Model Lineage")
        if best and best.is_successful:
            import hashlib
            data_hash = hashlib.sha256(
                pd.util.hash_pandas_object(df).values.tobytes()
            ).hexdigest()[:16]

            actual_preproc = best.trial.preprocess if best.trial.preprocess else preproc
            cfg = PipelineConfig(
                pipeline_id=f"endforecast_{report.task_type}_v1",
                task_type=report.task_type, preprocess=actual_preproc,
                features=best.trial.features,
                model={"type": best.trial.model, "params": best.trial.model_params},
                cv_strategy=best.trial.cv_strategy,
                refinement={"type": best.trial.refinement, "params": refinement_params},
                leak_rules=leak_rules,
            )
            pipe = Pipeline(cfg)
            try:
                pipe.fit(df, target_col=target_col, time_col=time_col)
                result.pipeline = pipe

                result.metric_tiers["training_data_hash"] = data_hash
                result.metric_tiers["pipeline_id"] = cfg.pipeline_id
                result.metric_tiers["model_family"] = best.trial.model
                result.metric_tiers["feature_list"] = best.trial.features
                result.metric_tiers["n_training_rows"] = len(df)
                result.metric_tiers["training_date_range"] = (
                    f"{df[time_col].min()} → {df[time_col].max()}"
                    if time_col and time_col in df.columns else "N/A"
                )

                raw_features = [c for c in df.columns if c not in {target_col, time_col, id_col, "date_time", "unique_id"} and not c.startswith("_")]
                engineered_features = [f for f in best.trial.features if f not in raw_features]
                result.metric_tiers["feature_dependencies"] = {
                    "raw_input_columns": raw_features,
                    "engineered_at_serving": engineered_features,
                    "must_be_available": raw_features + engineered_features,
                }

                logger.info("  Pipeline fitted successfully.")
                logger.info("  Lineage: hash=%s | %s rows | %s features",
                            data_hash, len(df), len(raw_features) + len(engineered_features))

                try:
                    from endforecast.engine.model_card import ModelCardGenerator
                    generator = ModelCardGenerator()
                    card = generator.generate(result)
                    card_path = Path("./endforecast_artifacts") / cfg.pipeline_id / "model_card.md"
                    card.save(card_path)
                    logger.info("  Model card: %s", card_path)
                except Exception as exc:
                    logger.debug("  Model card skipped: %s", exc)
            except Exception as exc:
                logger.error("  Pipeline fit failed: %s", exc)
                result.errors.append(str(exc))
        else:
            result.errors.append("No successful trial found.")

        # ═══ Phase 10: Drift Detection + Data Freshness ═════════
        logger.info("Phase 10: Drift Detection & Data Freshness")
        drift_report = DriftReport(summary_text="INITIAL_RUN")
        if time_col and time_col in df.columns:
            freshness_days, refresh_freq = DriftDetector.freshness_score(
                df, time_col=time_col,
            )
            drift_report.data_freshness_days = freshness_days
            drift_report.recommended_refresh_frequency = refresh_freq
            logger.info("  Freshness: %s days → %s refresh", freshness_days, refresh_freq)
            result.metric_tiers["data_freshness_days"] = freshness_days
            result.metric_tiers["recommended_refresh"] = refresh_freq
        result.drift_report = drift_report

        logger.info("=" * 50)
        logger.info(result.summary())
        return result

    # ── Internals ────────────────────────────────────────────────

    @staticmethod
    def _load_data(data: str | Path | pd.DataFrame) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data.copy()
        path = Path(data)
        if path.suffix in {".parquet", ".pqt"}:
            return pd.read_parquet(path)
        return pd.read_csv(path)

    @staticmethod
    def _plan_preprocessing(report: ExplorationReport) -> list[str]:
        """Task-agnostic baseline preprocessing.

        Only handles missing data. Model-specific preprocessing
        (scaling, etc.) is decided per-trial by the LLM planner.
        """
        steps: list[str] = []
        if report.missing_ratio > 0.05:
            steps.append("simple_impute")
        return steps

    def _execute_round(self, plan, df, target_col, time_col, round_num, preproc, hpo_method):
        results = []
        for trial in plan.trials:
            trial.round = round_num
            try:
                res = self._run_single_trial(trial, df, target_col, time_col, preproc, hpo_method, round_num)
                results.append(res)
            except Exception as exc:
                results.append(TrialResult(trial=trial, state=TrialState.FAILED, error_message=str(exc)))
        return results

    def _run_single_trial(self, trial, df, target_col, time_col, preproc, hpo_method, round_num=1):
        started = datetime.now()
        # ── Use trial-level preprocessing (LLM-decided per model),
        # ── falling back to global only if trial has none set ───
        actual_preproc = trial.preprocess if trial.preprocess else preproc
        logger.info("  trial %s: model=%s, features=%s, preproc=%s",
                     trial.id, trial.model, trial.features, actual_preproc)
        cfg = PipelineConfig(
            pipeline_id=trial.pipeline_id, task_type=trial.task_type,
            preprocess=actual_preproc, features=trial.features,
            model={"type": trial.model, "params": trial.model_params},
            cv_strategy=trial.cv_strategy, refinement={"type": trial.refinement},
        )
        pipe = Pipeline(cfg)
        # ── Vary random_state by round: each round tests on a different
        #     validation split; within the same round all trials share
        #     the same split for fair comparison.
        split_cfg = SplitConfig(method="train_test", test_size=0.2,
                                random_state=42 + round_num * 100)
        split_result = self.splitter.split(df, target_col=target_col, time_col=time_col, task_type=trial.task_type, config=split_cfg)
        if not split_result.folds:
            raise RuntimeError("No splits generated.")
        fold = split_result.folds[0]
        train_df, val_df = df.iloc[fold["train_idx"]], df.iloc[fold["val_idx"]]

        if hpo_method in {"grid", "random", "bayesian"}:
            hpo_cfg = HPOConfig(method=hpo_method, n_trials=5 if hpo_method == "bayesian" else 10)
            ps = _hpo_space(trial.model)

            def _obj(params):
                c2 = PipelineConfig(pipeline_id=trial.pipeline_id, task_type=trial.task_type, model={"type": trial.model, "params": params})
                p2 = Pipeline(c2)
                p2.fit(train_df, target_col=target_col, time_col=time_col)
                r = p2.predict(val_df, target_col=target_col, time_col=time_col)
                return r.metrics.get(trial.metric, float("inf"))

            hpo_res = self.hyperopt.optimize(_obj, ps, hpo_cfg)
            trial.model_params = hpo_res.best_params

        pipe.fit(train_df, target_col=target_col, time_col=time_col)
        pred_result = pipe.predict(val_df, target_col=target_col, time_col=time_col)
        y_val = val_df[target_col].values
        y_pred = pred_result.predictions

        # ── Align predictions & truth (lag features cause dropna → shorter preds) ──
        if len(y_pred) < len(y_val):
            logger.debug("  Aligning: preds=%s vs truth=%s (lag features dropped leading rows)",
                         len(y_pred), len(y_val))
            y_val = y_val[-len(y_pred):]

        mr = self.metrics_calc.evaluate(y_val, y_pred, task_type=trial.task_type, metrics=[trial.metric])

        # ── Stability assessment (multi-seed) ──────────────────
        stability_scores = [mr.value]
        for seed_offset in range(1, 3):
            try:
                scfg = SplitConfig(method="train_test", test_size=0.2,
                                   random_state=42 + seed_offset * 100)
                sr = self.splitter.split(df, target_col=target_col, time_col=time_col,
                                         task_type=trial.task_type, config=scfg)
                if not sr.folds:
                    continue
                f2 = sr.folds[0]
                tdf2 = df.iloc[f2["train_idx"]]
                vdf2 = df.iloc[f2["val_idx"]]
                p2 = Pipeline(PipelineConfig(
                    pipeline_id=f"{trial.pipeline_id}_seed{seed_offset}",
                    task_type=trial.task_type, preprocess=actual_preproc,
                    features=trial.features,
                    model={"type": trial.model, "params": trial.model_params},
                ))
                p2.fit(tdf2, target_col=target_col, time_col=time_col)
                pr2 = p2.predict(vdf2, target_col=target_col, time_col=time_col)
                yv2 = vdf2[target_col].values
                yp2 = pr2.predictions
                if len(yp2) < len(yv2):
                    yv2 = yv2[-len(yp2):]
                mr2 = self.metrics_calc.evaluate(yv2, yp2,
                                                 task_type=trial.task_type,
                                                 metrics=[trial.metric])
                stability_scores.append(mr2.value)
            except Exception as exc:
                logger.debug("Stability run %s failed: %s", seed_offset, exc)

        metric_mean = float(np.mean(stability_scores))
        metric_std = float(np.std(stability_scores, ddof=1)) if len(stability_scores) > 1 else 0.0

        return TrialResult(
            trial=trial, state=TrialState.COMPLETED,
            metric_value=metric_mean, metric_std=metric_std,
            metric_n_runs=len(stability_scores),
            metrics_detail={**{trial.metric: metric_mean}, **mr.auxiliary},
            started_at=started, completed_at=datetime.now(),
            duration_seconds=(datetime.now() - started).total_seconds(),
        )

    def _execute_ensemble(
        self,
        history: list[TrialResult],
        candidate_ids: list[str],
        df_train: pd.DataFrame,
        df_holdout: Optional[pd.DataFrame],
        target_col: str,
        time_col: Optional[str],
        preproc: list[str],
        method: str,
    ) -> Optional[np.ndarray]:
        """Train top-k models and combine their predictions on holdout data.

        Task-agnostic: works for classification, regression, time series.
        """
        if df_holdout is None or len(df_holdout) == 0:
            return None

        # Find candidate trials by id
        candidates = [r for r in history if r.trial.id in candidate_ids and r.is_successful]
        if len(candidates) < 2:
            logger.info("  Ensemble: need ≥2 candidates, found %d", len(candidates))
            return None

        all_preds: list[np.ndarray] = []
        for cand in candidates[:5]:  # Cap at 5 models
            try:
                actual_preproc = cand.trial.preprocess if cand.trial.preprocess else preproc
                cfg = PipelineConfig(
                    pipeline_id=f"ensemble_{cand.trial.id}",
                    task_type=cand.trial.task_type,
                    preprocess=actual_preproc,
                    features=cand.trial.features,
                    model={"type": cand.trial.model, "params": cand.trial.model_params},
                )
                pipe = Pipeline(cfg)
                pipe.fit(df_train, target_col=target_col, time_col=time_col)
                pr = pipe.predict(df_holdout, target_col=target_col, time_col=time_col)
                preds = pr.predictions
                all_preds.append(preds)
                logger.debug("  Ensemble candidate %s (%s) len=%d trained", cand.trial.id, cand.trial.model, len(preds))
            except Exception as exc:
                logger.debug("  Ensemble candidate %s failed: %s", cand.trial.id, exc)

        if len(all_preds) < 2:
            return None

        # ── Align: truncate all to shortest (lag features drop leading rows) ──
        min_len = min(len(p) for p in all_preds)
        aligned = [p[-min_len:] for p in all_preds]
        stack = np.column_stack(aligned)

        if method == "median_ensemble":
            return np.nanmedian(stack, axis=1)
        elif method == "mean_ensemble":
            return np.nanmean(stack, axis=1)
        elif method == "voting" and all_preds[0].dtype in (np.dtype(int), np.dtype(bool)):
            return (np.nanmean(stack, axis=1) >= 0.5).astype(float)
        else:
            # Default: median (robust to outliers)
            return np.nanmedian(stack, axis=1)


def _hpo_space(model: str) -> dict:
    return {
        "lightgbm": {"n_estimators": [100, 200, 500], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.05, 0.1]},
        "xgboost": {"n_estimators": [100, 200, 500], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.05, 0.1]},
        "random_forest": {"n_estimators": [100, 200, 500], "max_depth": [5, 10, 15]},
        "ridge": {"alpha": [0.01, 0.1, 1.0, 10.0]},
        "logistic": {"C": [0.01, 0.1, 1.0, 10.0]},
    }.get(model, {"n_estimators": [100, 200, 500]})