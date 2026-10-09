"""
Global configuration for EndForecast.

Configuration can be loaded from:
- Environment variables (prefix: ENDFORECAST_)
- YAML config file (endforecast.yaml)
- Programmatic API
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class LLMConfig:
    """LLM provider configuration."""

    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 120

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("ENDFORECAST_LLM_API_KEY")
        if self.api_base is None:
            self.api_base = os.environ.get("ENDFORECAST_LLM_API_BASE")


@dataclass
class StorageConfig:
    """Storage configuration for artifacts and results."""

    artifacts_dir: str = "./endforecast_artifacts"
    results_dir: str = "./endforecast_results"
    models_cache: str = "./endforecast_models"


@dataclass
class ExperimentConfig:
    """Experiment runtime configuration."""

    max_rounds: int = 10
    top_k: int = 5
    convergence_threshold: float = 0.01
    parallel_execution: bool = True
    max_workers: int = 4
    timeout_per_trial: int = 600  # seconds


@dataclass
class EndForecastConfig:
    """Root configuration for EndForecast."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    verbose: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EndForecastConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            EndForecastConfig instance populated from the file.
        """
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        llm = LLMConfig(**data.get("llm", {})) if data.get("llm") else LLMConfig()
        storage = StorageConfig(**data.get("storage", {})) if data.get("storage") else StorageConfig()
        experiment = ExperimentConfig(**data.get("experiment", {})) if data.get("experiment") else ExperimentConfig()
        return cls(llm=llm, storage=storage, experiment=experiment, verbose=data.get("verbose", False))

    @classmethod
    def from_env(cls) -> "EndForecastConfig":
        """Load configuration from environment variables.

        Environment variables:
            ENDFORECAST_LLM_PROVIDER, ENDFORECAST_LLM_MODEL, ENDFORECAST_LLM_API_KEY,
            ENDFORECAST_ARTIFACTS_DIR, ENDFORECAST_MAX_ROUNDS, etc.

        Returns:
            EndForecastConfig instance populated from environment.
        """
        llm = LLMConfig(
            provider=os.environ.get("ENDFORECAST_LLM_PROVIDER", "openai"),
            model=os.environ.get("ENDFORECAST_LLM_MODEL", "gpt-4o"),
            api_key=os.environ.get("ENDFORECAST_LLM_API_KEY"),
            api_base=os.environ.get("ENDFORECAST_LLM_API_BASE"),
        )
        storage = StorageConfig(
            artifacts_dir=os.environ.get("ENDFORECAST_ARTIFACTS_DIR", "./endforecast_artifacts"),
        )
        experiment = ExperimentConfig(
            max_rounds=int(os.environ.get("ENDFORECAST_MAX_ROUNDS", "10")),
            top_k=int(os.environ.get("ENDFORECAST_TOP_K", "5")),
        )
        return cls(llm=llm, storage=storage, experiment=experiment)