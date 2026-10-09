"""
CLI for EndForecast — The End of Forecast.

Usage:
    endforecast run --data sales.csv
    endforecast explore --data sales.csv
    endforecast deploy --config config.json --mode code
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="endforecast",
    help="The End of Forecast — AI agent-driven adaptive prediction automation",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    data: Path = typer.Option(..., "--data", "-d", help="Path to CSV/Parquet data file"),
    time_col: str = typer.Option("ds", "--time", "-t", help="Name of timestamp column"),
    target_col: str = typer.Option("y", "--target", "-y", help="Name of target column"),
    id_col: Optional[str] = typer.Option(None, "--id", help="Name of series ID column"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Directory to save results"),
    max_rounds: int = typer.Option(10, "--max-rounds", help="Maximum experiment rounds"),
    budget: str = typer.Option("auto", "--budget", help="Execution budget: fast, standard, deep, auto"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run a complete automated prediction pipeline."""
    import pandas as pd

    from endforecast.config import EndForecastConfig
    from endforecast.orchestrator import EndForecast
    from endforecast.agents.explorer import DataExplorer
    from endforecast.agents.planner import ExperimentPlanner
    from endforecast.engine.router import ExperimentRouter

    cfg = EndForecastConfig()
    cfg.verbose = verbose
    cfg.experiment.max_rounds = max_rounds

    console.print(Panel.fit(f"[bold blue]EndForecast[/bold blue] — The End of Forecast\nData: {data}", title="Starting"))

    if data.suffix in {".parquet", ".pqt"}:
        df = pd.read_parquet(data)
    else:
        df = pd.read_csv(data)
    console.print(f"  [green]{len(df):,}[/green] rows × [green]{len(df.columns)}[/green] cols")

    ef = EndForecast(config=cfg)
    result = ef.run(data=df, time_col=time_col, target_col=target_col, id_col=id_col, budget=budget)

    console.print(f"\n[bold]Result:[/bold] {'[green]Success[/green]' if result.success else '[red]Failed[/red]'}")
    if result.report:
        console.print(f"  Task: {result.report.task_type} | {result.report.summary()}")
    if result.best_trial:
        console.print(f"  Best: {result.best_trial.summary()}")
    if result.errors:
        for e in result.errors:
            console.print(f"  [red]Error:[/red] {e}")

    if result.success and output:
        result.export(mode="code", output_dir=output)
        console.print(f"[green]Exported to:[/green] {output}")


@app.command()
def explore(
    data: Path = typer.Option(..., "--data", "-d"),
    time_col: str = typer.Option("ds", "--time", "-t"),
    target_col: str = typer.Option("y", "--target", "-y"),
):
    """Run data exploration only — profile the dataset without training."""
    import pandas as pd
    from endforecast.agents.explorer import DataExplorer

    df = pd.read_parquet(data) if data.suffix in {".parquet", ".pqt"} else pd.read_csv(data)
    report = DataExplorer().explore(df, time_col=time_col, target_col=target_col)

    console.print(Panel(f"[bold cyan]{report.task_type.upper()}[/bold cyan] task detected"))
    console.print(f"  Rows: {report.n_rows:,}  Cols: {report.n_cols}")

    tbl = Table(title="Feature Fingerprint")
    tbl.add_column("Metric"); tbl.add_column("Value")
    for k, v in sorted(report.fingerprint.to_dict().items()):
        tbl.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    console.print(tbl)

    if report.quality_flags:
        console.print(f"\n[red]Flags:[/red] {', '.join(report.quality_flags)}")
    console.print(f"\n[dim]Models:[/dim] {', '.join(report.suggested_models)}")
    console.print(f"[dim]Metrics:[/dim] {', '.join(report.suggested_metrics)}")


@app.command()
def deploy(
    config_path: Path = typer.Option(..., "--config", "-c", help="Path to pipeline config.json"),
    mode: str = typer.Option("code", "--mode", "-m", help="Export mode: config, code, tool"),
    output: Path = typer.Option(Path("./endforecast_export"), "--output", "-o", help="Output directory"),
):
    """Deploy a trained pipeline from its config to a portable artifact."""
    from endforecast.engine.pipeline import PipelineConfig, Pipeline

    cfg = PipelineConfig.load(config_path)
    pipe = Pipeline(cfg)
    result_path = pipe.export(mode=mode, output_dir=output)
    console.print(f"[green]Deployed to:[/green] {result_path}")


@app.command()
def version():
    """Show version."""
    from endforecast import __version__
    console.print(f"EndForecast v{__version__} — The End of Forecast")


if __name__ == "__main__":
    app()