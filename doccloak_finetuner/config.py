"""Configuration schema for DocCloak.FineTuner."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class FineTuneConfig:
    task: str = "token-classification"
    base_model: str = "xlm-roberta-base"
    dataset: str = ""
    metric: str = "f1"
    metric_goal: str = "maximize"
    time_budget_per_run: str = "30m"
    device: str = "auto"
    languages: list[str] = field(default_factory=list)
    dataset_config: str | None = None
    text_column: str | None = None
    label_column: str | None = None
    max_samples: int | None = None
    eval_split: str = "test"

    # Multi-model support
    base_models: list[str] = field(default_factory=list)
    sweep_epochs: int = 3

    # Agent model (API cost vs quality)
    agent_model: str = "claude-sonnet-4-6"

    # Deployment constraints
    max_model_size_mb: int | None = None
    max_inference_ms: float | None = None

    # Safeguards
    max_runs: int = 40
    target_metric: float | None = None
    max_no_improvement: int = 8

    # QLoRA / text-generation specific
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_new_tokens: int = 1024

    @property
    def candidate_models(self) -> list[str]:
        """Ordered list of models to try. Falls back to base_model if base_models is empty."""
        return self.base_models if self.base_models else [self.base_model]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FineTuneConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @property
    def time_budget_seconds(self) -> int:
        budget = self.time_budget_per_run
        if budget.endswith("m"):
            return int(budget[:-1]) * 60
        if budget.endswith("h"):
            return int(budget[:-1]) * 3600
        if budget.endswith("s"):
            return int(budget[:-1])
        return int(budget)

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
