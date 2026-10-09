"""
Main orchestrator — complete 10-phase prediction pipeline per
the standard predictive modeling methodology.

Phases:
  0. Requirements Gathering
  1. Data Understanding (Exploration + EDA)
  2. Data Preparation (Preprocessing)
  3. LeakGuard Check
  4. Baseline Establishment ← NEW
  5. Experiment Planning & Execution
  6. Evaluation & Error Analysis
  7. Iterative Refinement (Diagnostician + Refiner)
  8. Interpretation & Explainability ← NEW
  9. Deployment
 10. Drift Detection ← NEW

All phases are task-agnostic (classification / regression / time series).
"""

from __future__ import annotations

import logging
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
        if self.explainability_result:
            lines.append(f"  Top Features: {[f for f, _ in self.explainability_result.top_features(3)]}")
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
    ) -> RunResult:
        result = RunResult()

        # ── Initialize optimization budget ────────────────────────
        opt_budget = OptimizationBudget(
            max_decisions=self.config.experiment.max_rounds * 3,
        )
        result.optimization_budget = opt_budget
        logger.info("Optimization budget: %s decisions available", opt_budget.max_decisions)

        # ═══ Phase 0: Requirements ═══════════════════════════════
        logger.info("=" * 50)
        logger.info("Phase 0: Requirements Gathering")
        if requirements:
            result.requirements = self.gatherer.gather(description=requirements)
            logger.info("  Task hint: %s", result.requirements.task_type_hint)

        # ═══ Phase 1: Data Understanding ══════════════════════════
        logger.info("Phase 1: Data Understanding")
        df = self._load_data(data)
        logger.info("  Loaded %s rows × %s cols", len(df), len(df.columns))

        report = self.explorer.explore(df, time_col=time_col, target_col=target_col, id_col=id_col)
        result.report = report
        # ── Diagnostic text for LLM agent (structured narrative, not isolated numbers)
        logger.info("  Diagnostic narrative:\n%s", report.fingerprint.diagnostic)
        if report.fingerprint.tags.get("is_cold_start"):
            result.metric_tiers["cold_start_detected"] = True
            result.metric_tiers["cold_start_groups"] = report.fingerprint.groups_below_threshold
        logger.info("  Task: %s | %s", report.task_type, report.summary())

        # ═══ Phase 2: Data Preparation ════════════════════════════
        logger.info("Phase 2: Data Preparation")
        preproc = self._plan_preprocessing(report)
        logger.info("  Steps: %s", preproc)

        # ═══ Phase 3: LeakGuard ══════════════════════════════════
        if leak_rules:
            logger.info("Phase 3: LeakGuard Check")
            guard = LeakGuard.from_config(leak_rules)
            lr = guard.check("check", df)
            result.leak_report = lr
            if not lr.all_clear:
                result.errors.append(f"LeakGuard BLOCKED: {lr.summary()}")
                return result
            logger.info("  PASSED")

        # ═══ Phase 4: Baseline Establishment ══════════════════════
        logger.info("Phase 4: Baseline Establishment")
        result.baseline_result = self.baseline_runner.run(
            df, target_col=target_col, time_col=time_col, id_col=id_col, task_type=report.task_type,
        )
        baseline_floor = result.baseline_result.baseline_floor
        logger.info("  Baseline floor: %.4f", baseline_floor)

        # ── Heuristic First check ───────────────────────────────
        # Based on Google Rules of ML #1-3: don't use ML if heuristics suffice.
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
        n_series = df[id_col].nunique() if (id_col and id_col in df.columns) else 1
        route = self.router.route(n_rows=len(df), n_series=n_series, budget_hint=budget)
        limits = self.router.get_resource_limits(route)
        plan = self.planner.plan(report, route=route, max_trials=limits["max_trials"])
        result.plan = plan
        logger.info("  Route: %s | Trials: %s", route.value, plan.n_trials)

        history: list[TrialResult] = []
        best: Optional[TrialResult] = None
        prev_best: Optional[float] = None

        for round_num in range(1, self.config.experiment.max_rounds + 1):
            if opt_budget.is_exhausted:
                logger.warning("  Optimization budget exhausted at round %s", round_num - 1)
                break

            logger.info("  --- Round %s/%s ---", round_num, self.config.experiment.max_rounds)
            round_results = self._execute_round(plan, df, target_col, time_col, round_num, preproc, limits["tuning"] if round_num > 1 else "none")
            if not round_results:
                logger.warning("    No successful trials.")
                continue
            opt_budget.consume(f"Round {round_num}: executed {len(round_results)} trials")
            history.extend(round_results)
            round_best = min(round_results, key=lambda r: r.metric_value)
            if best is None or round_best.metric_value < best.metric_value:
                best = round_best
                opt_budget.consume(f"Round {round_num}: new best model selected ({round_best.trial.model})")
            logger.info("    Best: %s %s=%.4f±%.4f (n=%s)", best.trial.pipeline_id, best.trial.metric.upper(), best.metric_value, best.metric_std, best.metric_n_runs)

            # Check baseline requirement
            if best.metric_value >= baseline_floor and baseline_floor != float("inf"):
                logger.info("    ⚠ Warning: Best model NOT better than baseline (%.4f)", baseline_floor)

            diagnosis = self.diagnostician.diagnose(round_results, baseline_metric=prev_best, previous_best=best.metric_value)
            if diagnosis:
                logger.info("    Diag: %s", diagnosis.summary)
                opt_budget.consume(f"Round {round_num}: diagnosis applied ({diagnosis.adjustment[:50]})")

            if prev_best is not None and best is not None and prev_best != 0:
                improvement = abs((best.metric_value - prev_best) / prev_best)
                if improvement < self.config.experiment.convergence_threshold:
                    logger.info("    Converged (impr %.2f%% < %.1f%%)", improvement * 100, self.config.experiment.convergence_threshold * 100)
                    opt_budget.consume(f"Round {round_num}: convergence declared")
                    break
            prev_best = best.metric_value

        result.history = history
        result.best_trial = best
        logger.info("  Trials completed: %s | Best: %.4f", len(history), best.metric_value if best else float("nan"))

        # ═══ Phase 6: Error Analysis + Data Snooping Tax ═══════════
        logger.info("Phase 6: Evaluation & Error Analysis")

        # Populate metric tiers
        result.metric_tiers = {
            "training_score": float("nan"),
            "validation_score": best.metric_value if best else float("nan"),
            "holdout_score": float("nan"),
            "optimization_rounds": len(history),
            "data_snooping_warning": False,
        }
        # Warn if many rounds of optimization on same validation data
        if len(history) > 5 and best and best.is_successful:
            result.metric_tiers["data_snooping_warning"] = True
            logger.info(
                "  ⚠ Data snooping warning: %s rounds of optimization on validation data. "
                "Validation score may be optimistic. Consider nested CV or hold-out.",
                len(history),
            )

        if best and best.is_successful and best.evaluation_df is not None:
            y_true = best.evaluation_df[target_col].values
            y_pred = best.metrics_detail.get("predictions", None)
            if y_pred is not None:
                mr = self.metrics_calc.evaluate(y_true, y_pred, task_type=report.task_type)
                logger.info("  %s", mr.interpretation)

        # ═══ Phase 7: Refinement ═════════════════════════════════
        refinement_params = {}
        if best and best.is_successful:
            logger.info("Phase 7: Iterative Refinement")
            try:
                # Method chosen by LLM if available, otherwise safe default.
                ref_method = "statistical"
                ref_result = self.refiner.refine(
                    np.array([best.metric_value]), report.task_type, method=ref_method,
                )
                logger.info("  Method: %s | Improvement: %.1f%%", ref_result.method, ref_result.improvement_pct)
                refinement_params = ref_result.details
            except Exception as exc:
                logger.warning("  Skipped: %s", exc)

        # ═══ Phase 8: Interpretation ═════════════════════════════
        logger.info("Phase 8: Interpretation & Explainability")
        if best and best.is_successful:
            try:
                # Build a quick pipeline to get fitted model + features
                cfg_temp = PipelineConfig(
                    pipeline_id="temp_explain", task_type=report.task_type,
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
            # ── Compute training data fingerprint for lineage ──
            import hashlib
            data_hash = hashlib.sha256(
                pd.util.hash_pandas_object(df).values.tobytes()
            ).hexdigest()[:16]

            cfg = PipelineConfig(
                pipeline_id=f"endforecast_{report.task_type}_v1",
                task_type=report.task_type, preprocess=preproc,
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

                # ── Model Lineage ─────────────────────────────
                result.metric_tiers["training_data_hash"] = data_hash
                result.metric_tiers["pipeline_id"] = cfg.pipeline_id
                result.metric_tiers["model_family"] = best.trial.model
                result.metric_tiers["feature_list"] = best.trial.features
                result.metric_tiers["n_training_rows"] = len(df)
                result.metric_tiers["training_date_range"] = (
                    f"{df[time_col].min()} → {df[time_col].max()}"
                    if time_col and time_col in df.columns else "N/A"
                )

                # ── Feature Store dependency list ─────────────
                # List all features the model depends on, so ops teams
                # know exactly what data must be available at serving time.
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

                # ── P0: Model Card Generation ──────────────────
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
        steps: list[str] = []
        if report.missing_ratio > 0.05:
            steps.append("simple_impute")
        if report.task_type == "timeseries":
            steps.append("standard_scaler")
        return steps

    def _execute_round(self, plan, df, target_col, time_col, round_num, preproc, hpo_method):
        results = []
        for trial in plan.trials:
            trial.round = round_num
            try:
                res = self._run_single_trial(trial, df, target_col, time_col, preproc, hpo_method)
                results.append(res)
            except Exception as exc:
                results.append(TrialResult(trial=trial, state=TrialState.FAILED, error_message=str(exc)))
        return results

    def _run_single_trial(self, trial, df, target_col, time_col, preproc, hpo_method):
        started = datetime.now()
        cfg = PipelineConfig(
            pipeline_id=trial.pipeline_id, task_type=trial.task_type,
            preprocess=preproc, features=trial.features,
            model={"type": trial.model, "params": trial.model_params},
            cv_strategy=trial.cv_strategy, refinement={"type": trial.refinement},
        )
        pipe = Pipeline(cfg)
        split_cfg = SplitConfig(method="train_test", test_size=0.2)
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
                r = p2.predict(val_df, time_col=time_col)
                return r.metrics.get(trial.metric, float("inf"))

            hpo_res = self.hyperopt.optimize(_obj, ps, hpo_cfg)
            trial.model_params = hpo_res.best_params

        pipe.fit(train_df, target_col=target_col, time_col=time_col)
        pred_result = pipe.predict(val_df, time_col=time_col)
        y_val = val_df[target_col].values
        mr = self.metrics_calc.evaluate(y_val, pred_result.predictions, task_type=trial.task_type, metrics=[trial.metric])

        # ── Stability assessment (multi-seed) ──────────────────
        stability_scores = [mr.value]
        for seed_offset in range(1, 3):  # Run 2 additional seeds
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
                    task_type=trial.task_type, preprocess=preproc,
                    features=trial.features,
                    model={"type": trial.model, "params": trial.model_params},
                ))
                p2.fit(tdf2, target_col=target_col, time_col=time_col)
                pr2 = p2.predict(vdf2, time_col=time_col)
                yv2 = vdf2[target_col].values
                mr2 = self.metrics_calc.evaluate(yv2, pr2.predictions,
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


def _hpo_space(model: str) -> dict:
    return {
        "lightgbm": {"n_estimators": [100, 200, 500], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.05, 0.1]},
        "xgboost": {"n_estimators": [100, 200, 500], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.05, 0.1]},
        "random_forest": {"n_estimators": [100, 200, 500], "max_depth": [5, 10, 15]},
        "ridge": {"alpha": [0.01, 0.1, 1.0, 10.0]},
    }.get(model, {"n_estimators": [100, 200, 500]})