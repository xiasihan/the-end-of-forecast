"""
EndForecast — The End of Forecast.

An AI agent-driven adaptive prediction automation platform that
implements the standard 10-phase predictive modeling methodology
for classification, regression, and time series problems.

Phases: Requirements → Understanding → Preparation → LeakGuard →
Baselines → Experimentation → Evaluation → Refinement →
Interpretation → Deployment → Monitoring.
"""

__version__ = "0.1.0"
__author__ = "EndForecast Contributors"

from endforecast.orchestrator import EndForecast, RunResult
from endforecast.engine.pipeline import Pipeline, PipelineConfig, PipelineResult
from endforecast.engine.leakguard import LeakGuard, LeakRule, LeakReport, LeakSeverity
from endforecast.engine.fingerprint import FeatureFingerprint, FingerprintCalculator
from endforecast.engine.router import ExperimentRouter, Route
from endforecast.engine.trial import Trial, TrialResult, TrialState, OptimizationBudget
from endforecast.engine.baseline import BaselineRunner, BaselineResult
from endforecast.engine.explainability import Explainer, ExplainabilityResult
from endforecast.engine.drift import DriftDetector, DriftReport
from endforecast.agents.requirements import RequirementsGatherer, RequirementsSpec
from endforecast.agents.explorer import DataExplorer, ExplorationReport
from endforecast.agents.planner import ExperimentPlanner, ExperimentPlan
from endforecast.agents.diagnostician import Diagnostician, DiagnosisReport
from endforecast.agents.refiner import Refiner, RefinementResult
from endforecast.deployment.pipeline_exporter import PipelineExporter
from endforecast.deployment.code_generator import CodeGenerator
from endforecast.deployment.tool_generator import ToolGenerator
from endforecast.evaluation.splitter import DataSplitter, SplitConfig, SplitResult, NestedCVSplitter, NestedCVConfig, NestedCVResult
from endforecast.evaluation.metrics import MetricCalculator, MetricResult, StatisticalComparison, ComparisonResult, StabilityEvaluator, StabilityResult, CalibrationMetrics, CalibrationResult
from endforecast.evaluation.hyperopt import HPOptimizer, HPOConfig, HPOResult

__all__ = [
    "EndForecast", "RunResult",
    "Pipeline", "PipelineConfig", "PipelineResult",
    "LeakGuard", "LeakRule", "LeakReport", "LeakSeverity",
    "FeatureFingerprint", "FingerprintCalculator",
    "ExperimentRouter", "Route",
    "Trial", "TrialResult", "TrialState", "OptimizationBudget",
    "BaselineRunner", "BaselineResult",
    "Explainer", "ExplainabilityResult",
    "DriftDetector", "DriftReport",
    "RequirementsGatherer", "RequirementsSpec",
    "DataExplorer", "ExplorationReport",
    "ExperimentPlanner", "ExperimentPlan",
    "Diagnostician", "DiagnosisReport",
    "Refiner", "RefinementResult",
    "PipelineExporter", "CodeGenerator", "ToolGenerator",
    "DataSplitter", "SplitConfig", "SplitResult",
    "NestedCVSplitter", "NestedCVConfig", "NestedCVResult",
    "MetricCalculator", "MetricResult",
    "StatisticalComparison", "ComparisonResult",
    "StabilityEvaluator", "StabilityResult",
    "CalibrationMetrics", "CalibrationResult",
    "HPOptimizer", "HPOConfig", "HPOResult",
]