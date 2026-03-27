"""Tests for config parsing."""

from pathlib import Path

from doccloak_finetuner.config import FineTuneConfig


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(content)
    return p


class TestFineTuneConfig:
    def test_minimal_config(self, tmp_path):
        path = _write_yaml(tmp_path, """
task: token-classification
base_model: xlm-roberta-base
dataset: test/dataset
metric: f1
""")
        cfg = FineTuneConfig.from_yaml(path)
        assert cfg.task == "token-classification"
        assert cfg.base_model == "xlm-roberta-base"
        assert cfg.metric == "f1"
        assert cfg.metric_goal == "maximize"

    def test_full_config(self, tmp_path):
        path = _write_yaml(tmp_path, """
task: token-classification
base_model: xlm-roberta-base
dataset: ai4privacy/open-pii-masking-500k-ai4privacy
metric: f1
metric_goal: maximize
time_budget_per_run: 24h
device: auto
text_column: mbert_tokens
label_column: mbert_token_classes
eval_split: validation
""")
        cfg = FineTuneConfig.from_yaml(path)
        assert cfg.metric_goal == "maximize"
        assert cfg.time_budget_per_run == "24h"
        assert cfg.text_column == "mbert_tokens"
        assert cfg.label_column == "mbert_token_classes"
        assert cfg.eval_split == "validation"
        assert cfg.device == "auto"

    def test_time_budget_seconds(self, tmp_path):
        path = _write_yaml(tmp_path, """
task: token-classification
base_model: bert-base
dataset: test/data
metric: f1
time_budget_per_run: 24h
""")
        cfg = FineTuneConfig.from_yaml(path)
        assert cfg.time_budget_seconds == 86400

    def test_minimize_goal(self, tmp_path):
        path = _write_yaml(tmp_path, """
task: token-classification
base_model: bert-base
dataset: test/data
metric: loss
metric_goal: minimize
""")
        cfg = FineTuneConfig.from_yaml(path)
        assert cfg.metric_goal == "minimize"

    def test_safeguards_defaults(self, tmp_path):
        path = _write_yaml(tmp_path, """
task: token-classification
base_model: bert-base
dataset: test/data
metric: f1
""")
        cfg = FineTuneConfig.from_yaml(path)
        assert cfg.max_runs == 40
        assert cfg.target_metric is None
        assert cfg.max_no_improvement == 8
