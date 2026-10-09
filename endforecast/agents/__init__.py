"""
Agent layer — the intelligence core of EndForecast.

Agent pipeline (in execution order):
  0. RequirementsGatherer — elicit prediction requirements
  1. DataExplorer — task detection, data profiling, fingerprint
  1b. PreprocessingPlanner — data cleaning & feature engineering
  2. ExperimentPlanner — multi-model experiment design
  3. Diagnostician — failure pattern analysis & strategy adjustment
  4. Refiner — post-processing for additional accuracy gains

All agents are task-agnostic and work identically for classification,
regression, and time series forecasting.
"""

from endforecast.agents.requirements import RequirementsGatherer, RequirementsSpec
from endforecast.agents.explorer import DataExplorer, ExplorationReport
from endforecast.agents.planner import ExperimentPlanner, ExperimentPlan
from endforecast.agents.diagnostician import Diagnostician, DiagnosisReport
from endforecast.agents.refiner import Refiner, RefinementResult

__all__ = [
    "RequirementsGatherer",
    "RequirementsSpec",
    "DataExplorer",
    "ExplorationReport",
    "ExperimentPlanner",
    "ExperimentPlan",
    "Diagnostician",
    "DiagnosisReport",
    "Refiner",
    "RefinementResult",
]