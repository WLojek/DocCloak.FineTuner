<h1 align="center">DocCloak.FineTuner</h1>

<p align="center">Agentic fine-tuning for NER/PII detection.<br>Part of the DocCloak ecosystem.<br>Inspired by <a href="https://github.com/karpathy/autoresearch">autoresearch</a> by Andrej Karpathy.</p>

Define your candidate models, dataset, and constraints in YAML. DocCloak.FineTuner uses the Anthropic API to drive an autonomous research loop — the agent systematically explores the hyperparameter space, compares model architectures, and converges on the best F1 score. All experiments are logged and git-committed.

The agent operates through validated tools with hard constraints — it cannot use unlisted models, set hyperparameters outside allowed ranges, or bypass safeguards.

## Features

- **Autonomous hyperparameter optimization** — Claude agent explores LR, batch size, sequence length, regularization
- **Multi-model sweep** — compare multiple base models in a single run
- **Hard constraint enforcement** — model whitelist, HP ranges, and safeguards enforced at API/orchestrator level
- **Live training output** — watch progress, loss, and metrics in real-time
- **ONNX export** — convert trained models for cross-platform / browser deployment
- **INT8 quantization** — reduce model size by ~75% for edge deployment
- **GPU cloud ready** — automated setup script for RunPod, Vast.ai, and other GPU providers

## How It Works

```
.env (API key) + config.yaml
  -> doccloak-finetune init -c config.yaml -o workspace
    -> Generates workspace: train.py, prepare.py, guard.py, results.tsv
  -> doccloak-finetune run -c config.yaml -o workspace
    -> Starts API loop with Claude agent:
        1. Agent proposes experiment via set_hyperparameters tool (validated)
        2. Orchestrator writes train.py and runs training
        3. Training output streams live (progress, loss, metrics)
        4. Results returned: F1, precision, recall, inference speed, model size
        5. Agent analyzes, decides next experiment
        6. Repeat until safeguards trigger or agent calls finish()
```

Progressive phases:
- **Phase 0**: Architecture sweep — quick test of all candidate models
- **Phase 1**: Learning rate search on top models
- **Phase 2**: Batch size & sequence length optimization
- **Phase 3**: Regularization & training dynamics
- **Phase 4**: Refinement around the best configuration

## Requirements

- Python 3.11+
- [Anthropic API key](https://console.anthropic.com/) (for the `run` command only — `init` and `export` don't need it)
- GPU recommended (CUDA or Apple MPS) — CPU works but is slow

## Supported Models

Any model from [HuggingFace Hub](https://huggingface.co/models) compatible with `AutoModelForTokenClassification`. This includes:

- **BERT** variants (bert-base, bert-large, etc.)
- **RoBERTa** / **XLM-RoBERTa** (multilingual)
- **HerBERT** (Polish)
- **DeBERTa** / **DeBERTa-v3**
- **DistilBERT** (lightweight)
- **ELECTRA**, **ALBERT**, and more

## Supported Datasets

- **Local datasets** — HuggingFace `datasets` format on disk (e.g. `./datasets/polish-pii`)
- **HuggingFace Hub** — any NER dataset by ID (e.g. `ai4privacy/open-pii-masking-500k`)

Datasets must have a text column (list of tokens) and a label column (list of BIO tags).

## Installation

### Local (Mac / Linux)

```bash
git clone https://github.com/WLojek/DocCloak.FineTuner.git
cd DocCloak.FineTuner
pip install -e .
```

That's it — PyTorch will use Apple MPS on Mac or CPU on Linux automatically.

### GPU Cloud (RunPod, Vast.ai, etc.)

On cloud GPU machines, use the setup script instead — it creates a virtual environment and installs the correct PyTorch build for your GPU's CUDA version:

```bash
cd DocCloak.FineTuner
bash setup_gpu.sh
source .venv/bin/activate
```

> **Note:** `setup_gpu.sh` is only needed on cloud GPU machines. On your local Mac or Linux, `pip install -e .` is sufficient.

## Quick Start

### 1. Set your API key

```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

### 2. Prepare your dataset

Place your HuggingFace-format dataset in `datasets/`. The dataset should have `tokens` (list of strings) and `ner_tags` (list of BIO tag strings) columns.


### 3. Create a config

```yaml
# config.yaml
base_models:
  - allegro/herbert-base-cased       # Polish
  - xlm-roberta-base                 # Multilingual

dataset: ./datasets/polish-pii       # Local path or HuggingFace Hub ID
task: token-classification
text_column: tokens
label_column: ner_tags
eval_split: validation

agent_model: claude-sonnet-4-6       # or claude-opus-4-6 for best quality
metric: f1
metric_goal: maximize
sweep_epochs: 3
time_budget_per_run: 10h
max_runs: 20
max_no_improvement: 4
```

See `config.herbert.yaml` for a complete example.

### 4. Initialize workspace

```bash
doccloak-finetune init -c config.yaml -o workspace
```

This generates the training scripts without starting the API loop — useful for inspection or manual training.

### 5. Run autonomous research loop

```bash
doccloak-finetune run -c config.yaml -o workspace
```

The orchestrator will systematically test models and hyperparameters, logging results to `workspace/results.tsv`.

### 6. Export the trained model

After training, export the best model for deployment:

```bash
# PyTorch format (default)
doccloak-finetune export -i workspace/best_model -o export --pytorch

# ONNX format (for cross-platform deployment)
doccloak-finetune export -i workspace/best_model -o export --onnx

# ONNX with INT8 dynamic quantization (for browser / edge)
doccloak-finetune export -i workspace/best_model -o export --onnx --int8 dynamic

# ONNX with INT8 static quantization (more accurate, needs calibration data)
doccloak-finetune export -i workspace/best_model -o export --onnx --int8 static --calibration-data datasets/polish-pii

# All formats at once
doccloak-finetune export -i workspace/best_model -o export --pytorch --onnx --int8 dynamic
```

For ONNX export, install the optional dependencies:

```bash
pip install -e ".[onnx]"
```

### Export output

```
export/
├── onnx/
│   ├── model.onnx              # Full FP32 ONNX model (~473 MB)
│   ├── model_quantized.onnx    # INT8 quantized model (~119 MB, 75% smaller)
│   ├── config.json             # Model config (label map, architecture)
│   ├── tokenizer.json          # Tokenizer data
│   ├── tokenizer_config.json   # Tokenizer config
│   └── export_metadata.json    # Export info (format, quantization, sizes)
└── pytorch/                    # Only if --pytorch flag used
    ├── model.safetensors
    ├── config.json
    └── tokenizer files...
```

The quantized ONNX model is ready for browser deployment (via ONNX Runtime WebAssembly) or any cross-platform runtime.

## GPU Cloud Deployment (RunPod)

### 1. Create a pod

- GPU: RTX 4090 (24 GB VRAM) or similar
- Container disk: 20 GB
- Volume disk: 50 GB
- Enable SSH terminal access

### 2. Upload and setup

From your local machine:

```bash
scp -P <PORT> -i ~/.ssh/id_ed25519 -r DocCloak.FineTuner root@<HOST>:/workspace/
```

On the pod:

```bash
cd /workspace/DocCloak.FineTuner
bash setup_gpu.sh
source .venv/bin/activate
echo "ANTHROPIC_API_KEY=your-key" > .env
doccloak-finetune init -c config.herbert.yaml -o workspace_herbert
doccloak-finetune run -c config.herbert.yaml -o workspace_herbert
```

### 3. Download results

From your local machine:

```bash
scp -P <PORT> -i ~/.ssh/id_ed25519 -r root@<HOST>:/workspace/DocCloak.FineTuner/workspace_herbert/best_model ./best_model
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `doccloak-finetune init -c config.yaml -o workspace` | Generate training workspace |
| `doccloak-finetune run -c config.yaml -o workspace` | Run autonomous research loop |
| `doccloak-finetune export -i model_dir -o export_dir [flags]` | Export trained model |

### Export flags

| Flag | Description |
|------|-------------|
| `--pytorch` | Export as PyTorch model |
| `--onnx` | Export as ONNX model |
| `--int8 dynamic` | Apply INT8 dynamic quantization (no calibration needed) |
| `--int8 static` | Apply INT8 static quantization (requires `--calibration-data`) |
| `--calibration-data PATH` | Dataset for static quantization calibration |

## Constraint Enforcement

| Constraint | Level | How |
|---|---|---|
| Model whitelist | API (hard) | `enum` in tool schema — API rejects invalid models |
| LR scheduler | API (hard) | `enum` in tool schema |
| LR range (1e-6 to 1e-3) | Orchestrator (hard) | Python validation before execution |
| Batch size (4 to 64) | Orchestrator (hard) | Python validation before execution |
| Epoch range (1 to 30) | Orchestrator (hard) | Python validation before execution |
| Max runs / no improvement | guard.py + orchestrator (hard) | Checked after each run; blocks further experiments |
| Max model size / latency | Orchestrator (hard) | Checked after each training run |

## Safeguards

| Safeguard | Default | What it does |
|---|---|---|
| `max_runs` | 40 | Hard limit on total experiments |
| `max_no_improvement` | 8 | Stop after N consecutive runs with no improvement |
| `target_metric` | none | Stop when metric reaches this value |
| `max_model_size_mb` | none | Reject models exceeding this size |
| `max_inference_ms` | none | Reject models slower than this |

## Config Reference

```yaml
# Models to compare (agent sweeps all, then deep-tunes the best)
base_models:
  - xlm-roberta-base
  - bert-base-multilingual-cased

# Or single model
# base_model: xlm-roberta-base

# Dataset (local path or HuggingFace Hub ID)
dataset: ./datasets/my-ner-data
task: token-classification
text_column: tokens
label_column: ner_tags
eval_split: validation
dataset_config: null
max_samples: null
languages: []

# Agent model (cost vs quality)
agent_model: claude-sonnet-4-6   # ~$5-8/run | claude-opus-4-6 ~$15-20/run

# Optimization
metric: f1
metric_goal: maximize
sweep_epochs: 2                  # Quick epochs for Phase 0 model sweep
time_budget_per_run: 20m         # Per-experiment time limit
device: auto

# Deployment constraints (optional)
max_model_size_mb: null
max_inference_ms: null

# Safeguards
max_runs: 25
max_no_improvement: 6
target_metric: null
```

## Project Structure

```
DocCloak.FineTuner/
├── doccloak_finetuner/
│   ├── cli.py                 # CLI: init, run, export commands
│   ├── config.py              # Config dataclass + YAML parsing
│   ├── orchestrator.py        # API-driven research loop
│   ├── exporter.py            # ONNX export + INT8 quantization
│   ├── tools.py               # Tool definitions + validation
│   └── scaffold/
│       ├── generator.py       # Renders templates into workspace
│       └── templates/
│           ├── token_classification_train.py.j2
│           ├── token_classification_prepare.py.j2
│           └── guard.py.j2
├── tests/
│   ├── test_cli.py            # Workspace generation tests
│   └── test_config.py         # Config parsing tests
├── config.herbert.yaml        # Example config for Polish PII
├── setup_gpu.sh               # GPU cloud setup script
├── .env.example               # API key template
├── pyproject.toml             # Package config + dependencies
└── LICENSE                    # AGPL-3.0
```

## Security

- **Never commit `.env`** — it contains your API key. Use `.env.example` as a template.
- The `.gitignore` excludes `.env`, datasets, model weights, and generated workspaces.
- The orchestrator enforces hard constraints on all agent actions — the agent cannot bypass validation or execute arbitrary code.

## License

AGPL-3.0 — see [LICENSE](LICENSE).
