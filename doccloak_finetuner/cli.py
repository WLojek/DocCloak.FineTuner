"""CLI entry point for DocCloak.FineTuner."""

import os
import subprocess
from pathlib import Path

import click

from .config import FineTuneConfig


def _generate(config: FineTuneConfig, output_dir: Path) -> None:
    """Generate train.py, prepare.py, and guard.py from templates."""
    from .scaffold.generator import generate_experiment
    generate_experiment(config, output_dir)


def _init_workspace(output_dir: Path) -> None:
    """Create results.tsv and git init."""
    results_path = output_dir / "results.tsv"
    if not results_path.exists():
        results_path.write_text("run\ttag\tphase\tmodel\tf1\tprecision\trecall\t"
                                "loss\tparams_M\tsize_mb\tinference_ms\tvalue\t"
                                "duration_s\tnotes\n")

    if not (output_dir / ".git").exists():
        subprocess.run(["git", "init"], cwd=output_dir, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=output_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "doccloak-finetune: workspace initialized"],
            cwd=output_dir, capture_output=True,
        )


@click.group()
@click.version_option()
def main():
    """DocCloak.FineTuner — Autonomous fine-tuning for DocCloak PII models."""
    pass


@main.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config.yaml")
@click.option("--output", "-o", default="./workspace", type=click.Path(), help="Output directory")
def init(config, output):
    """Generate workspace from config (no API key needed)."""
    cfg = FineTuneConfig.from_yaml(config)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    _generate(cfg, output_path)
    _init_workspace(output_path)

    click.echo(f"Workspace scaffolded in {output_path}/")
    click.echo("  - train.py       (training script)")
    click.echo("  - prepare.py     (data loading — do not modify)")
    click.echo("  - guard.py       (safeguards — do not modify)")
    click.echo("  - results.tsv    (experiment log)")
    click.echo()
    click.echo("To run the autonomous research loop:")
    click.echo(f"  doccloak-finetune run -c {config}")


@main.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config.yaml")
@click.option("--output", "-o", default="./workspace", type=click.Path(), help="Workspace directory")
def run(config, output):
    """Run autonomous fine-tuning research loop via Anthropic API."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv is optional if ANTHROPIC_API_KEY is already set

    if not os.environ.get("ANTHROPIC_API_KEY"):
        click.echo("Error: ANTHROPIC_API_KEY not set.")
        click.echo("Set it in .env or as an environment variable.")
        click.echo("See .env.example for the format.")
        raise SystemExit(1)

    cfg = FineTuneConfig.from_yaml(config)

    from .orchestrator import Orchestrator
    orchestrator = Orchestrator(cfg, Path(output))
    orchestrator.run()


@main.command()
@click.option("--input", "-i", required=True, type=click.Path(exists=True), help="Path to trained model (e.g. workspace/best_model)")
@click.option("--output", "-o", required=True, type=click.Path(), help="Output directory")
@click.option("--task", type=click.Choice(["token-classification", "text-generation", "auto"]), default="auto", help="Model task type (auto-detects from model config)")
@click.option("--pytorch", is_flag=True, help="Export as PyTorch model")
@click.option("--onnx", is_flag=True, help="Export as ONNX model")
@click.option("--gptq", is_flag=True, help="Export with GPTQ INT4 quantization (requires GPU, text-generation only)")
@click.option("--int8", "int8_mode", type=click.Choice(["dynamic", "static"]), default=None, help="Apply INT8 quantization: dynamic (no calibration) or static (uses eval data)")
@click.option("--int4", is_flag=True, help="Apply INT4 quantization (text-generation only)")
@click.option("--calibration-data", type=click.Path(exists=True), default=None, help="Dataset path for static INT8 calibration (required with --int8 static)")
def export(input, output, task, pytorch, onnx, gptq, int8_mode, int4, calibration_data):
    """Export trained model to specified format with optional quantization.

    Examples:

      doccloak-finetune export -i best_model -o export --pytorch

      doccloak-finetune export -i best_model -o export --onnx

      doccloak-finetune export -i best_model -o export --onnx --int8 dynamic

      doccloak-finetune export -i best_model -o export --onnx --int4 --task text-generation

      doccloak-finetune export -i best_model -o export --pytorch --onnx --int8 dynamic
    """
    model_dir = Path(input)
    output_dir = Path(output)

    # Auto-detect task type
    if task == "auto":
        # Check if it's a LoRA/causal model
        adapter_config = model_dir / "adapter_config.json"
        if adapter_config.exists():
            task = "text-generation"
        else:
            import json
            config_path = model_dir / "config.json"
            if config_path.exists():
                model_config = json.loads(config_path.read_text())
                architectures = model_config.get("architectures", [])
                if any("CausalLM" in a for a in architectures):
                    task = "text-generation"
                else:
                    task = "token-classification"
            else:
                task = "token-classification"
        click.echo(f"Auto-detected task: {task}")

    # Default to pytorch if nothing specified
    if not pytorch and not onnx and not gptq:
        pytorch = True

    if int8_mode and not onnx:
        raise click.ClickException("--int8 requires --onnx")

    if int4 and not onnx:
        raise click.ClickException("--int4 requires --onnx")

    if int8_mode == "static" and not calibration_data:
        raise click.ClickException("--int8 static requires --calibration-data")

    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Exporting model from {model_dir}")
    flags = []
    if pytorch:
        flags.append("PyTorch")
    if onnx:
        flags.append("ONNX")
    if gptq:
        flags.append("GPTQ INT4")
    if int8_mode:
        flags.append(f"INT8 {int8_mode}")
    if int4:
        flags.append("INT4")
    click.echo(f"  Task: {task}")
    click.echo(f"  Formats: {', '.join(flags)}")
    click.echo()

    if task == "text-generation":
        from .exporter_causal import export_causal_model
        export_causal_model(
            model_dir=model_dir,
            output_dir=output_dir,
            do_onnx=onnx,
            do_int4=int4,
            do_gptq=gptq,
            do_pytorch=pytorch,
        )
    else:
        from .exporter import export_pytorch, export_onnx, validate_onnx, _validate_model_dir
        _validate_model_dir(model_dir)
        quantize = int8_mode or "none"

        if pytorch:
            export_pytorch(model_dir, output_dir)

        if onnx:
            cal_path = Path(calibration_data) if calibration_data else None
            onnx_dir = export_onnx(model_dir, output_dir, quantize=quantize, calibration_data=cal_path)
            click.echo()
            validate_onnx(onnx_dir)

    click.echo()
    click.echo("Export complete!")
