"""Code generator — produces a standalone Python prediction module
with embedded refinement logic from the winning pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from endforecast.engine.pipeline import Pipeline

logger = logging.getLogger(__name__)


class CodeGenerator:
    def __init__(self, pipeline: "Pipeline") -> None:
        self.pipeline = pipeline

    def export(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        models_dir = output_dir / "models"
        models_dir.mkdir(exist_ok=True)

        import joblib
        joblib.dump(self.pipeline._model, models_dir / "model.bin")
        for name, artifact in self.pipeline._artifacts.items():
            joblib.dump(artifact, models_dir / f"{name}.pkl")

        (output_dir / "predictor.py").write_text(self._generate_code(), encoding="utf-8")
        (output_dir / "requirements.txt").write_text("pandas>=2.0\nnumpy>=1.24\nscikit-learn>=1.3\njoblib>=1.3\n")
        self._write_readme(output_dir)
        return output_dir

    def _generate_code(self) -> str:
        cfg = self.pipeline.config
        refine_code = self._generate_refinement_code()
        return dedent(f'''"""
EndForecast Auto-Generated Predictor
Pipeline: {cfg.pipeline_id}
Task: {cfg.task_type}
"""

import joblib
import numpy as np
import pandas as pd

class Predictor:
    def __init__(self, model_dir: str = "./models"):
        import os
        self.model = joblib.load(os.path.join(model_dir, "model.bin"))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict(df)
        return self._refine(raw)

    def _refine(self, raw: np.ndarray) -> np.ndarray:
{self._indent(refine_code, 8)}

if __name__ == "__main__":
    import sys
    p = Predictor()
    data = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "new_data.csv")
    print(p.predict(data))
''')

    def _generate_refinement_code(self) -> str:
        """Generate the _refine() method body from the pipeline's refinement config.

        Reads PipelineConfig.refinement → extracts method + serialized params →
        produces the corresponding Python code. If no refinement is configured,
        returns a simple pass-through.
        """
        refinement = self.pipeline.config.refinement
        if not refinement or refinement.get("type", "none") == "none":
            return "return raw"

        params: dict[str, Any] = refinement.get("params", {})
        method = params.get("method", "none")

        if method == "residual_correction":
            bias = params.get("bias_correction", 0)
            return f"return raw + {bias}"

        if method == "threshold_optimization":
            t = params.get("optimized_threshold", 0.5)
            return f"return (raw >= {t}).astype(float)"

        if method == "statistical":
            bias = params.get("bias_correction", 0)
            lo = params.get("clip_lower", float("-inf"))
            hi = params.get("clip_upper", float("inf"))
            if bias == 0 and lo == float("-inf") and hi == float("inf"):
                return "return raw"
            lines = []
            if bias != 0:
                lines.append(f"corrected = raw + {bias}")
            else:
                lines.append("corrected = raw")
            lines.append(f"return np.clip(corrected, {lo}, {hi})")
            return "\n".join(lines)

        if method == "none":
            return "return raw"

        logger.warning("Unknown refinement method '%s' — skipping code generation.", method)
        return "return raw"

    @staticmethod
    def _indent(code: str, spaces: int) -> str:
        prefix = " " * spaces
        return "\n".join(prefix + line if line.strip() else line for line in code.split("\n"))

    def _write_readme(self, output_dir: Path) -> None:
        cfg = self.pipeline.config
        refine_note = ""
        if cfg.refinement and cfg.refinement.get("type", "none") != "none":
            method = cfg.refinement.get("params", {}).get("method", cfg.refinement.get("type", ""))
            refine_note = f"\n- Refinement: {method} (embedded in `_refine()`)"

        (output_dir / "README.md").write_text(
            f"# {cfg.pipeline_id}\n\n"
            f"Auto-generated predictor by EndForecast.\n\n"
            f"- Task: {cfg.task_type}\n"
            f"- Model: {cfg.model.get('type', 'unknown')}{refine_note}\n\n"
            "```bash\npip install -r requirements.txt\npython predictor.py new_data.csv\n```\n"
        )