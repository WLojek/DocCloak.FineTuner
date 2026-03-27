"""Tool definitions and validation for the autonomous research agent."""

from __future__ import annotations

from dataclasses import dataclass

from .config import FineTuneConfig

# Hyperparameter bounds — enforced in Python before execution
HP_BOUNDS = {
    "learning_rate": (1e-6, 1e-3),
    "batch_size": (4, 64),
    "num_epochs": (1, 30),
    "warmup_ratio": (0.0, 0.3),
    "weight_decay": (0.0, 0.2),
    "freeze_layers": (0, 8),
}


@dataclass
class Hyperparameters:
    base_model: str
    learning_rate: float
    batch_size: int
    num_epochs: int
    warmup_ratio: float
    weight_decay: float
    max_seq_length: int
    freeze_layers: int
    gradient_accumulation_steps: int
    lr_scheduler: str
    hypothesis: str


def validate_hyperparameters(hp: dict, config: FineTuneConfig) -> list[str]:
    """Validate hyperparameters against hard bounds. Returns list of violations."""
    errors = []

    if hp.get("base_model") not in config.candidate_models:
        errors.append(
            f"model '{hp.get('base_model')}' not in allowed list: {config.candidate_models}"
        )

    for param, (low, high) in HP_BOUNDS.items():
        val = hp.get(param)
        if val is not None and not (low <= val <= high):
            errors.append(f"{param}={val} outside allowed range [{low}, {high}]")

    if hp.get("max_seq_length") not in (128, 192, 256, 384, 512):
        errors.append(
            f"max_seq_length={hp.get('max_seq_length')} not in [128, 192, 256, 384, 512]"
        )

    if hp.get("gradient_accumulation_steps") not in (1, 2, 4, 8):
        errors.append(
            f"gradient_accumulation_steps={hp.get('gradient_accumulation_steps')} not in [1, 2, 4, 8]"
        )

    if hp.get("lr_scheduler") not in ("linear", "cosine", "cosine_with_restarts"):
        errors.append(
            f"lr_scheduler='{hp.get('lr_scheduler')}' not in [linear, cosine, cosine_with_restarts]"
        )

    return errors


def build_tool_definitions(config: FineTuneConfig) -> list[dict]:
    """Build tool definitions with config-specific enums for the API."""
    models = config.candidate_models

    return [
        {
            "name": "set_hyperparameters",
            "description": (
                "Set hyperparameters for the next training experiment. "
                "You MUST call this before run_experiment. "
                "Include a hypothesis explaining what you're testing and why."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "base_model": {
                        "type": "string",
                        "enum": models,
                        "description": "Base model to fine-tune",
                    },
                    "learning_rate": {
                        "type": "number",
                        "description": "Learning rate (1e-6 to 1e-3)",
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Training batch size (4 to 64)",
                    },
                    "num_epochs": {
                        "type": "integer",
                        "description": "Number of training epochs (1 to 30)",
                    },
                    "warmup_ratio": {
                        "type": "number",
                        "description": "Warmup ratio (0.0 to 0.3)",
                    },
                    "weight_decay": {
                        "type": "number",
                        "description": "Weight decay (0.0 to 0.2)",
                    },
                    "max_seq_length": {
                        "type": "integer",
                        "enum": [128, 192, 256, 384, 512],
                        "description": "Maximum sequence length for tokenization",
                    },
                    "freeze_layers": {
                        "type": "integer",
                        "description": "Number of encoder layers to freeze (0 to 8)",
                    },
                    "gradient_accumulation_steps": {
                        "type": "integer",
                        "enum": [1, 2, 4, 8],
                        "description": "Gradient accumulation steps",
                    },
                    "lr_scheduler": {
                        "type": "string",
                        "enum": ["linear", "cosine", "cosine_with_restarts"],
                        "description": "Learning rate scheduler type",
                    },
                    "hypothesis": {
                        "type": "string",
                        "description": (
                            "What you are testing and why. "
                            "E.g. 'Testing higher LR (5e-5) because previous run at 2e-5 "
                            "showed slow convergence with rising F1 trend.'"
                        ),
                    },
                },
                "required": [
                    "base_model",
                    "learning_rate",
                    "batch_size",
                    "num_epochs",
                    "warmup_ratio",
                    "weight_decay",
                    "max_seq_length",
                    "freeze_layers",
                    "gradient_accumulation_steps",
                    "lr_scheduler",
                    "hypothesis",
                ],
            },
        },
        {
            "name": "run_experiment",
            "description": (
                "Execute training with the current hyperparameters. "
                "Returns F1, precision, recall, loss, model size, inference speed, "
                "and any constraint violations. "
                "You must call set_hyperparameters first."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "read_results",
            "description": (
                "Read the experiment history from results.tsv. "
                "Use this to understand what has been tried and plan your next experiment."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "last_n": {
                        "type": "integer",
                        "description": "Number of most recent rows to return. 0 = all.",
                    },
                },
                "required": ["last_n"],
            },
        },
        {
            "name": "finish",
            "description": (
                "End the research loop and declare the best model. "
                "Call this when you've found the best configuration or "
                "when further experiments are unlikely to improve F1 significantly."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "selected_model": {
                        "type": "string",
                        "description": "HuggingFace model ID of the best model",
                    },
                    "f1": {
                        "type": "number",
                        "description": "Best F1 score achieved",
                    },
                    "rationale": {
                        "type": "string",
                        "description": (
                            "Why this model was selected. Include trade-off analysis: "
                            "what was sacrificed (if anything) vs alternatives."
                        ),
                    },
                },
                "required": ["selected_model", "f1", "rationale"],
            },
        },
    ]
