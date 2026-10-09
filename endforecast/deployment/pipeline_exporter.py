"""
Pipeline config exporter — serializes the winning pipeline to a
portable JSON/YAML configuration file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from endforecast.engine.pipeline import Pipeline

logger = logging.getLogger(__name__)


class PipelineExporter:
    """Export a trained Pipeline as a portable configuration package."""

    def __init__(self, pipeline: "Pipeline") -> None:
        self.pipeline = pipeline

    def export(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir = output_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        self.pipeline.config.save(output_dir / "config.json")
        import joblib
        joblib.dump(self.pipeline._model, artifacts_dir / "model.bin")
        for name, artifact in self.pipeline._artifacts.items():
            joblib.dump(artifact, artifacts_dir / f"{name}.pkl")

        self._write_readme(output_dir)
        logger.info("Exported pipeline to %s", output_dir)
        return output_dir

    def _write_readme(self, output_dir: Path) -> None:
        cfg = self.pipeline.config
        (output_dir / "README.md").write_text(
            f"# {cfg.pipeline_id}\n\n"
            f"Task: {cfg.task_type} | Model: {cfg.model.get('type', 'unknown')}\n\n"
            "```python\nfrom endforecast import PipelineConfig, Pipeline\n"
            "cfg = PipelineConfig.load('config.json')\n"
            "pipe = Pipeline(cfg)\nresult = pipe.predict(new_data)\n```\n"
        )