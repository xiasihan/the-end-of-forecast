"""Core engine — prediction pipeline, leak detection, fingerprint extraction,
experiment routing, trial management, baselines, explainability, and drift detection."""

from endforecast.engine.pipeline import Pipeline, PipelineConfig, PipelineResult
from endforecast.engine.leakguard import LeakGuard, LeakRule, LeakReport, LeakSeverity
from endforecast.engine.fingerprint import FeatureFingerprint, FingerprintCalculator
from endforecast.engine.router import ExperimentRouter, Route
from endforecast.engine.trial import Trial, TrialResult, TrialState, OptimizationBudget
from endforecast.engine.baseline import BaselineRunner, BaselineResult
from endforecast.engine.explainability import Explainer, ExplainabilityResult
from endforecast.engine.drift import DriftDetector, DriftReport, TrainingServingSkewReport
from endforecast.engine.model_card import ModelCard, ModelCardGenerator
from endforecast.engine.label_noise import LabelNoiseDetector, LabelNoiseReport

__all__ = [
    "Pipeline", "PipelineConfig", "PipelineResult",
    "LeakGuard", "LeakRule", "LeakReport", "LeakSeverity",
    "FeatureFingerprint", "FingerprintCalculator",
    "ExperimentRouter", "Route",
    "Trial", "TrialResult", "TrialState", "OptimizationBudget",
    "BaselineRunner", "BaselineResult",
    "Explainer", "ExplainabilityResult",
    "DriftDetector", "DriftReport", "TrainingServingSkewReport",
    "ModelCard", "ModelCardGenerator",
    "LabelNoiseDetector", "LabelNoiseReport",
]