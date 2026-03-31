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

## Supported Tasks

### Token Classification (NER / PII Detection)

Fine-tunes encoder models (BERT-family) on BIO-labeled NER datasets.

**Supported models:** Any `AutoModelForTokenClassification` from HuggingFace — BERT, RoBERTa, XLM-RoBERTa, HerBERT, DeBERTa, DistilBERT, ELECTRA, ALBERT, etc.

**Dataset format:** Text column (list of tokens) + label column (list of BIO tags).

**Metrics:** F1, precision, recall.

### Text Generation (Scribe / Document Generation)

Fine-tunes decoder models (LLaMA-family) with QLoRA for template-filling document generation.

**Supported models:** Any `AutoModelForCausalLM` from HuggingFace — LLaMA, Mistral, etc.

**Dataset format:** Input column (template + user request) + output column (filled document).

**Metrics:** ROUGE-L, slot accuracy, perplexity.

**Extra dependencies:** `pip install -e ".[qlora]"` (peft, bitsandbytes, trl, rouge-score).

## Supported Datasets

- **Local datasets** — HuggingFace `datasets` format on disk (e.g. `./datasets/polish-pii`, `./datasets/scribe-no`)
- **HuggingFace Hub** — any dataset by ID (e.g. `ai4privacy/open-pii-masking-500k`)

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

See `config.herbert.yaml` for a complete NER example and `config.scribe.yaml` for text generation.

### Scribe (Text Generation) Config

```yaml
base_models:
  - NbAiLab/nb-llama-3.2-1B

dataset: ./datasets/scribe-no
task: text-generation
text_column: input
label_column: output
eval_split: validation

agent_model: claude-sonnet-4-6
metric: rouge_l
metric_goal: maximize
sweep_epochs: 1
time_budget_per_run: 4h
max_runs: 25
max_no_improvement: 5

# QLoRA hyperparameters
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
max_new_tokens: 1024
```

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

After training, export the best model for deployment.

#### Token Classification (NER) Export

```bash
# PyTorch format (default)
doccloak-finetune export -i workspace/best_model -o export --pytorch

# ONNX format (for browser / cross-platform)
doccloak-finetune export -i workspace/best_model -o export --onnx

# ONNX + INT8 dynamic quantization (~75% smaller, no GPU needed)
doccloak-finetune export -i workspace/best_model -o export --onnx --int8 dynamic

# ONNX + INT8 static quantization (best accuracy, needs calibration data)
doccloak-finetune export -i workspace/best_model -o export --onnx --int8 static --calibration-data datasets/polish-pii

# Multiple formats at once
doccloak-finetune export -i workspace/best_model -o export --pytorch --onnx --int8 dynamic
```

#### Text Generation (Scribe) Export

For Scribe models, the export automatically detects and merges LoRA adapters into the base model.

```bash
# PyTorch (merged model, ~5GB)
doccloak-finetune export -i workspace_scribe/best_model -o export_scribe --pytorch

# ONNX + INT8 quantization (~1.4GB, no GPU needed)
doccloak-finetune export -i workspace_scribe/best_model -o export_scribe --onnx

# GPTQ INT4 quantization (~500MB, REQUIRES GPU — run on RunPod)
doccloak-finetune export -i workspace_scribe/best_model -o export_scribe --gptq

# Force task type if auto-detection fails
doccloak-finetune export -i workspace_scribe/best_model -o export_scribe --gptq --task text-generation
```

#### Export Format Comparison

| Format | Size (1B model) | Quality loss | GPU required | Best for |
|--------|-----------------|-------------|-------------|----------|
| PyTorch (safetensors) | ~5 GB | None | No | Server deployment, further conversion |
| ONNX FP32 | ~5.7 GB | None | No | Cross-platform inference |
| ONNX INT8 | ~1.4 GB | <1% | No | Browser (acceptable size) |
| **GPTQ INT4** | **~500 MB** | **<1%** | **Yes** | **Browser (recommended)** |

**For browser deployment, use GPTQ INT4.** It produces the smallest model with negligible quality loss. Run it on RunPod (~$0.10 for 10 minutes).

#### Dependencies

```bash
# ONNX export only
pip install -e ".[onnx]"

# QLoRA training + GPTQ export + ONNX (everything)
pip install -e ".[qlora,onnx]"
```

On GPU cloud (RunPod), `setup_gpu.sh` installs all dependencies automatically.

#### Export Output Structure

```
export/
├── pytorch/                    # --pytorch
│   ├── model.safetensors
│   ├── config.json
│   └── tokenizer files...
├── onnx/                       # --onnx
│   ├── model.onnx
│   ├── model.onnx_data         # External weights (large models)
│   └── config + tokenizer...
├── onnx_quantized/             # --onnx --int8
│   ├── model_quantized.onnx
│   └── config + tokenizer...
├── gptq_int4/                  # --gptq (requires GPU)
│   ├── model.safetensors       # ~500MB quantized weights
│   ├── config.json
│   ├── quantize_config.json    # GPTQ parameters
│   └── tokenizer files...
├── merged/                     # Auto-created when merging LoRA
│   └── (full merged model)
└── export_metadata.json        # Export info
```

## GPU Cloud Deployment (RunPod)

### 1. Create a pod

- GPU: RTX 4090 (24 GB VRAM) — sufficient for 1B models with QLoRA
- Container disk: 20 GB
- Volume disk: 50 GB (workspace persists across restarts)
- Use "SSH over exposed TCP" for SCP file transfers (regular SSH doesn't support SCP)

### 2. Upload and setup

From your local machine (use the TCP port from RunPod UI):

```bash
# Copy FineTuner code
scp -P <PORT> -i ~/.ssh/id_ed25519 -r DocCloak.FineTuner root@<HOST>:/workspace/DocCloak.FineTuner

# Copy dataset (for Scribe)
scp -P <PORT> -i ~/.ssh/id_ed25519 -r DocCloak.DataForge/output_scribe/legal-hr/no/hf_scribe_dataset root@<HOST>:/workspace/DocCloak.FineTuner/datasets/scribe-no

# SSH in
ssh root@<HOST> -p <PORT> -i ~/.ssh/id_ed25519
```

On the pod:

```bash
cd /workspace/DocCloak.FineTuner
bash setup_gpu.sh
source .venv/bin/activate
echo "ANTHROPIC_API_KEY=your-key" > .env
```

### 3. Run training

```bash
# NER (token classification)
doccloak-finetune run -c config.herbert.yaml -o workspace_herbert

# Scribe (text generation)
doccloak-finetune run -c config.scribe.yaml -o workspace_scribe
```

Use `tmux` to keep it running after disconnecting:

```bash
tmux new -s training
doccloak-finetune run -c config.scribe.yaml -o workspace_scribe
# Detach: Ctrl+B then D
# Reconnect later: tmux attach -t training
```

### 4. Export and download

```bash
# GPTQ INT4 — smallest model for browser (~500MB, recommended)
doccloak-finetune export -i workspace_scribe/best_model -o export_scribe --gptq

# Or ONNX INT8 (~1.4GB, no GPTQ dependency needed)
export TMPDIR=/workspace/tmp && mkdir -p /workspace/tmp
doccloak-finetune export -i workspace_scribe/best_model -o export_scribe --onnx

# Or PyTorch only (for further conversion later)
doccloak-finetune export -i workspace_scribe/best_model -o export_scribe --pytorch
```

From your local machine:

```bash
# Download GPTQ model (~500MB)
scp -P <PORT> -i ~/.ssh/id_ed25519 -r root@<HOST>:/workspace/DocCloak.FineTuner/export_scribe/gptq_int4 ./export_scribe/gptq_int4

# Or download everything
scp -P <PORT> -i ~/.ssh/id_ed25519 -r root@<HOST>:/workspace/DocCloak.FineTuner/export_scribe ./export_scribe
```

Then stop the pod to save money.

### Quick GPTQ-only export (if model is already trained)

If you already have a merged model and just need GPTQ quantization:

```bash
# From local machine — copy merged model to RunPod
ssh root@<HOST> -p <PORT> -i ~/.ssh/id_ed25519 "mkdir -p /workspace/export_scribe"
scp -P <PORT> -i ~/.ssh/id_ed25519 -r export_scribe/merged root@<HOST>:/workspace/export_scribe/merged

# SSH in and run GPTQ
ssh root@<HOST> -p <PORT> -i ~/.ssh/id_ed25519
cd /workspace/DocCloak.FineTuner
source .venv/bin/activate
doccloak-finetune export -i /workspace/export_scribe/merged -o /workspace/export_scribe --gptq

# Download result (~500MB, ~2 min)
# From local: scp -P <PORT> -i ~/.ssh/id_ed25519 -r root@<HOST>:/workspace/export_scribe/gptq_int4 ./export_scribe/gptq_int4
```

This takes ~10 minutes on an RTX 4090 and costs ~$0.10.

### Troubleshooting (RunPod)

| Problem | Solution |
|---------|----------|
| `OSError: No space left on device` during ONNX export | Set `export TMPDIR=/workspace/tmp && mkdir -p /workspace/tmp` — the container `/tmp` is only 20GB, use the workspace volume instead |
| `README.md not found` during `pip install -e .` | Run `touch README.md` — SCP may skip hidden/dotfiles |
| `RuntimeError: element 0 of tensors does not require grad` | Template fix needed: `model.enable_input_require_grads()` must be called after `get_peft_model()` for QLoRA + gradient checkpointing |
| Nested directory after SCP (`DocCloak.FineTuner/DocCloak.FineTuner/`) | Flatten: `mv DocCloak.FineTuner/DocCloak.FineTuner/* DocCloak.FineTuner/ && rm -rf DocCloak.FineTuner/DocCloak.FineTuner` |
| `datasets/` empty after SCP | Re-copy: `scp -P <PORT> -i ~/.ssh/id_ed25519 -r <dataset_path> root@<HOST>:/workspace/DocCloak.FineTuner/datasets/scribe-no` |
| Training killed after SSH disconnect | Use `tmux` (see step 3 above) |
| `top_p` warnings during generation eval | Harmless — the model ignores unsupported generation flags |
| Tied weights warning during ONNX export | Harmless — embedding/output weight sharing is normal for LLMs |
| `RuntimeError: GPTQ requires CUDA GPU` | GPTQ must run on a GPU machine. Use RunPod or similar. Cannot run on CPU/Mac |
| `CUDA initialization: NVIDIA driver too old` | The system torch has wrong CUDA version. Run `bash setup_gpu.sh` which installs correct torch for your GPU |
| `Failed to serialize proto` during ONNX export | Model too large for protobuf. Use `--no-post-process` (handled automatically in our exporter) or use `--gptq` instead |
| ONNX INT4 larger than INT8 | Normal — ONNX Runtime's naive INT4 only quantizes MatMul weights. Use `--gptq` for proper INT4 (~500MB) |

## CLI Reference

| Command | Description |
|---------|-------------|
| `doccloak-finetune init -c config.yaml -o workspace` | Generate training workspace |
| `doccloak-finetune run -c config.yaml -o workspace` | Run autonomous research loop |
| `doccloak-finetune export -i model_dir -o export_dir [flags]` | Export trained model |

### Export flags

| Flag | Description | Requires |
|------|-------------|----------|
| `--pytorch` | Export as PyTorch safetensors | Nothing |
| `--onnx` | Export as ONNX model | `pip install -e ".[onnx]"` |
| `--gptq` | GPTQ INT4 quantization (~500MB, best for browser) | GPU + `pip install -e ".[qlora]"` |
| `--int8 dynamic` | ONNX INT8 dynamic quantization | `--onnx` |
| `--int8 static` | ONNX INT8 static quantization | `--onnx` + `--calibration-data` |
| `--int4` | ONNX INT4 quantization (naive, less effective than GPTQ) | `--onnx` |
| `--task TYPE` | Force task type: `token-classification`, `text-generation`, `auto` | Nothing |
| `--calibration-data PATH` | Dataset for static INT8 calibration | `--int8 static` |

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

### Token Classification (NER)

```yaml
base_models:
  - xlm-roberta-base
  - allegro/herbert-base-cased

dataset: ./datasets/polish-pii      # Local path or HuggingFace Hub ID
task: token-classification
text_column: tokens                  # Column with list of word tokens
label_column: ner_tags               # Column with list of BIO tags
eval_split: validation
dataset_config: null                 # HuggingFace dataset config name
max_samples: null                    # Limit dataset size (for quick tests)
languages: []                        # Filter by language column (if dataset has one)

agent_model: claude-sonnet-4-6       # Agent that drives the research loop
metric: f1                           # Primary metric to optimize
metric_goal: maximize
sweep_epochs: 3                      # Quick epochs for Phase 0 model sweep
time_budget_per_run: 10h             # Max time per experiment
device: auto                         # cuda, mps, cpu, or auto

max_model_size_mb: null              # Reject models exceeding this size
max_inference_ms: null               # Reject models slower than this
max_runs: 25                         # Hard limit on experiments
max_no_improvement: 6                # Stop after N runs with no improvement
target_metric: null                  # Stop when metric reaches this value
```

### Text Generation (Scribe)

```yaml
base_models:
  - NbAiLab/nb-llama-3.2-1B

dataset: ./datasets/scribe-no       # HuggingFace Dataset with input/output columns
task: text-generation
text_column: input                   # Column with template + user request
label_column: output                 # Column with filled document
eval_split: validation

agent_model: claude-sonnet-4-6
metric: rouge_l                      # Primary metric (ROUGE-L for generation quality)
metric_goal: maximize
sweep_epochs: 1                      # 1 epoch per experiment (fast iteration)
time_budget_per_run: 4h
device: auto
max_runs: 25
max_no_improvement: 5

# QLoRA hyperparameters (text-generation only, ignored for token-classification)
lora_rank: 16                        # LoRA adapter rank (4-64)
lora_alpha: 32                       # LoRA scaling factor (usually 2x rank)
lora_dropout: 0.05                   # LoRA dropout (0.0-0.3)
max_new_tokens: 1024                 # Max tokens to generate during evaluation
```

## Project Structure

```
DocCloak.FineTuner/
├── doccloak_finetuner/
│   ├── cli.py                 # CLI: init, run, export commands
│   ├── config.py              # Config dataclass + YAML parsing
│   ├── orchestrator.py        # API-driven research loop
│   ├── exporter.py            # ONNX export + INT8 quantization (token classification)
│   ├── exporter_causal.py     # ONNX export + LoRA merge + INT4 (text generation)
│   ├── tools.py               # Tool definitions + validation
│   └── scaffold/
│       ├── generator.py       # Renders templates into workspace
│       └── templates/
│           ├── token_classification_train.py.j2
│           ├── token_classification_prepare.py.j2
│           ├── text_generation_train.py.j2
│           ├── text_generation_prepare.py.j2
│           └── guard.py.j2
├── tests/
│   ├── test_cli.py            # Workspace generation tests
│   └── test_config.py         # Config parsing tests
├── config.herbert.yaml        # Example config for Polish PII (token classification)
├── config.scribe.yaml         # Example config for Scribe (text generation)
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
