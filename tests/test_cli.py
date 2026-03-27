"""Tests for CLI and workspace generation."""


import pytest

from doccloak_finetuner.cli import _generate, _init_workspace
from doccloak_finetuner.config import FineTuneConfig


def _scaffold(config, tmp_path):
    """Helper: generate workspace."""
    _generate(config, tmp_path)
    _init_workspace(tmp_path)


@pytest.fixture
def ml_config():
    return FineTuneConfig(
        task="token-classification",
        base_model="xlm-roberta-base",
        dataset="test/dataset",
        metric="f1",
        text_column="tokens",
        label_column="labels",
    )


class TestGenerateWorkspace:
    def test_creates_train_py(self, tmp_path, ml_config):
        _scaffold(ml_config, tmp_path)
        assert (tmp_path / "train.py").exists()

    def test_creates_prepare_py(self, tmp_path, ml_config):
        _scaffold(ml_config, tmp_path)
        assert (tmp_path / "prepare.py").exists()

    def test_creates_results_tsv(self, tmp_path, ml_config):
        _scaffold(ml_config, tmp_path)
        results = (tmp_path / "results.tsv").read_text()
        assert "run\ttag\tphase\tmodel\tf1\tprecision\trecall" in results

    def test_inits_git(self, tmp_path, ml_config):
        _scaffold(ml_config, tmp_path)
        assert (tmp_path / ".git").exists()

    def test_creates_guard_py(self, tmp_path, ml_config):
        _scaffold(ml_config, tmp_path)
        assert (tmp_path / "guard.py").exists()

    def test_no_duplicate_results_tsv(self, tmp_path, ml_config):
        _scaffold(ml_config, tmp_path)
        (tmp_path / "results.tsv").write_text("run\tmetric\tvalue\tduration_s\tnotes\n1\tf1\t0.5\t10\ttest\n")
        _scaffold(ml_config, tmp_path)
        content = (tmp_path / "results.tsv").read_text()
        assert "test" in content
