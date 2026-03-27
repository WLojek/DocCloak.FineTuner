"""Generates train.py, prepare.py, and guard.py from config and templates."""

from pathlib import Path
from jinja2 import Environment, FileSystemLoader

from ..config import FineTuneConfig

TEMPLATES_DIR = Path(__file__).parent / "templates"


def generate_experiment(config: FineTuneConfig, output_dir: Path) -> None:
    # Resolve relative dataset paths to absolute so they work from workspace/
    if config.dataset and not config.dataset.startswith(("/", "http")):
        config.dataset = str(Path(config.dataset).resolve())

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        keep_trailing_newline=True,
    )

    template_map = {
        "token-classification": "token_classification",
        "text-classification": "text_classification",
        "seq2seq": "seq2seq",
    }

    task_key = template_map.get(config.task)
    if not task_key:
        raise ValueError(f"Unsupported task: {config.task}. Supported: {list(template_map.keys())}")

    # Generate train.py
    train_template = env.get_template(f"{task_key}_train.py.j2")
    train_content = train_template.render(config=config)
    (output_dir / "train.py").write_text(train_content)

    # Generate prepare.py
    prepare_template = env.get_template(f"{task_key}_prepare.py.j2")
    prepare_content = prepare_template.render(config=config)
    (output_dir / "prepare.py").write_text(prepare_content)

    # Generate guard.py
    guard_template = env.get_template("guard.py.j2")
    guard_content = guard_template.render(config=config)
    (output_dir / "guard.py").write_text(guard_content)

    # Initialize results.tsv
    results_path = output_dir / "results.tsv"
    if not results_path.exists():
        results_path.write_text(
            "run\ttag\tphase\tmodel\tf1\tprecision\trecall\t"
            "loss\tparams_M\tsize_mb\tinference_ms\tvalue\t"
            "duration_s\tnotes\n"
        )
