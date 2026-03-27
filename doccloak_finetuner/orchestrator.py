"""API-driven orchestrator for autonomous PII model fine-tuning."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import time
from io import StringIO
from pathlib import Path

import anthropic
import click

from .config import FineTuneConfig
from .tools import Hyperparameters, build_tool_definitions, validate_hyperparameters

RESULTS_HEADER = [
    "run", "tag", "phase", "model", "f1", "precision", "recall",
    "loss", "params_M", "size_mb", "inference_ms", "value", "duration_s", "notes",
]


def build_system_prompt(config: FineTuneConfig) -> str:
    """Build the system prompt that replaces program.md."""
    models_str = ", ".join(config.candidate_models)
    size_limit = f"{config.max_model_size_mb} MB" if config.max_model_size_mb else "none"
    speed_limit = f"{config.max_inference_ms} ms/sample" if config.max_inference_ms else "none"

    return f"""You are an autonomous ML research agent fine-tuning PII (Personally Identifiable Information) detection models for DocCloak — a document anonymization tool.

Your goal: maximize F1 score through systematic, hypothesis-driven experimentation.

## Setup
- Candidate models: {models_str}
- Sweep epochs (Phase 0): {config.sweep_epochs}
- Full training time budget per run: {config.time_budget_per_run}
- Max model size: {size_limit}
- Max inference latency: {speed_limit}
- Safeguards: max {config.max_runs} runs, stop after {config.max_no_improvement} consecutive runs with no improvement

## Protocol

You have 4 tools:
1. `set_hyperparameters` — configure the next experiment (model, LR, batch size, etc.)
2. `run_experiment` — execute training and get results (F1, speed, size)
3. `read_results` — inspect experiment history
4. `finish` — declare the best model and end

Every experiment: set_hyperparameters → run_experiment → analyze results → decide next.

## Progressive Phases

### Phase 0: Architecture Sweep
For EACH candidate model, run a quick experiment ({config.sweep_epochs} epochs) with safe defaults:
- LR=2e-5, BS=16, epochs={config.sweep_epochs}, seq_len=256, no freezing
- Record F1, speed, and size for each model
- Eliminate models that violate constraints (size/speed limits)
- Select top 2-3 models for deep tuning

### Phase 1: Learning Rate Search (per top model)
Test: 1e-5, 2e-5, 5e-5, 1e-4. Find the optimal LR range.
This has the highest impact on F1.

### Phase 2: Batch Size & Sequence Length
Test BS: 8, 16, 32. Test seq_len: 128, 256, 384, 512.
PII entities appear at various positions — longer sequences capture more context.
Class imbalance (non-PII >> PII): smaller batches may help.

### Phase 3: Regularization & Training Dynamics
Test warmup_ratio, weight_decay, lr_scheduler, freeze_layers, gradient_accumulation.
Prevent overfitting and improve generalization.

### Phase 4: Refinement
Fine-grained tuning around the best configuration found so far.
Small adjustments to LR, epochs, regularization.

## Decision Logic

After each experiment:
- F1 improved significantly (>0.01): double down on this direction
- F1 improved slightly (0.001-0.01): keep, try related changes
- F1 got worse: try something orthogonal
- Flat for 3+ runs: switch to a different phase or model
- Stuck for 5+ runs: try fundamentally different approach (different model, 10x LR change)

## Multi-Objective Priority

1. **F1 is primary** — always maximize F1
2. **Hard constraints first** — reject models exceeding size/speed limits
3. **Tiebreak (F1 within 0.02)** — prefer faster inference, then smaller size
4. When finishing, explicitly state trade-offs between models

## PII-Specific Notes

- Class imbalance (non-PII >> PII tokens): smaller batches, more epochs may help
- Multilingual data: layer freezing preserves pre-trained multilingual knowledge
- Rare entity types (passports, SSNs): longer sequences, more epochs for repeated exposure
- F1 = harmonic mean of precision and recall. Both matter for safe PII detection.

## Model-Specific Notes

- **DistilBERT** (distilbert-base-multilingual-cased): 40% fewer params, faster. Higher LR (3e-5 to 5e-5). Only 6 layers.
- **mBERT** (bert-base-multilingual-cased): Standard BERT. Similar size to XLM-R base but generally lower multilingual quality.
- **XLM-RoBERTa base** (xlm-roberta-base): Strong multilingual baseline. LR 1e-5 to 3e-5.
- **XLM-RoBERTa large** (xlm-roberta-large): Best quality but 3x size/speed. Lower LR (5e-6 to 1e-5), freeze more layers.

## Rules

- ALWAYS state your hypothesis before each experiment (in the hypothesis field)
- ALWAYS call set_hyperparameters before run_experiment
- Use read_results to track what you've tried — don't repeat failed experiments
- If a run fails (OOM, crash), try smaller settings (lower BS, shorter seq_len)
- Call finish when you're confident you've found the best configuration
"""


class Orchestrator:
    """Drives the autonomous research loop via the Anthropic API."""

    def __init__(self, config: FineTuneConfig, workspace_dir: Path):
        self.config = config
        self.workspace = workspace_dir
        self.client = anthropic.Anthropic()
        self.tools = build_tool_definitions(config)
        self.system_prompt = build_system_prompt(config)
        self.model = config.agent_model
        self.messages: list[dict] = []
        self.current_hp: Hyperparameters | None = None
        self.run_count = 0
        self.best_f1 = 0.0
        self.best_model = ""
        self.finished = False
        self.guard_triggered = False

    def run(self) -> None:
        """Execute the full research loop."""
        self._setup_workspace()
        self._init_results_tsv()

        click.echo("=" * 60)
        click.echo("DocCloak.FineTuner — Autonomous Research Loop")
        click.echo(f"Agent: {self.model}")
        click.echo(f"Models: {', '.join(self.config.candidate_models)}")
        click.echo(f"Max runs: {self.config.max_runs}")
        click.echo(f"Budget per run: {self.config.time_budget_per_run}")
        click.echo("=" * 60)

        self.messages = [
            {
                "role": "user",
                "content": (
                    "Start the autonomous research loop. "
                    "Begin with Phase 0: sweep all candidate models with default hyperparameters. "
                    "Use read_results first to check if there are any prior experiments."
                ),
            }
        ]

        while not self.finished:
            try:
                self._step()
            except anthropic.RateLimitError as e:
                retry_after = int(e.response.headers.get("retry-after", "60"))
                click.echo(f"\n[RATE LIMITED] Waiting {retry_after}s...")
                time.sleep(retry_after)
            except anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    click.echo(f"\n[API ERROR {e.status_code}] Retrying in 30s...")
                    time.sleep(30)
                else:
                    raise

        click.echo("\n" + "=" * 60)
        click.echo("Research loop complete.")
        click.echo(f"Best F1: {self.best_f1:.4f} ({self.best_model})")
        click.echo("=" * 60)

    def _step(self) -> None:
        """Single step: call API, execute tool calls, feed results back."""
        # System prompt with cache_control — cached after first call (90% cheaper)
        system = [
            {
                "type": "text",
                "text": self.system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        # Use compaction for long conversations to prevent context overflow
        # and reduce input token costs on later calls
        use_compaction = self.model in ("claude-opus-4-6", "claude-sonnet-4-6")

        if use_compaction:
            with self.client.beta.messages.stream(
                betas=["compact-2026-01-12"],
                model=self.model,
                max_tokens=16000,
                system=system,
                thinking={"type": "adaptive"},
                tools=self.tools,
                messages=self.messages,
                cache_control={"type": "ephemeral"},
                context_management={"edits": [{"type": "compact_20260112"}]},
            ) as stream:
                response = stream.get_final_message()
        else:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=16000,
                system=system,
                tools=self.tools,
                messages=self.messages,
                cache_control={"type": "ephemeral"},
            ) as stream:
                response = stream.get_final_message()

        # Print any text output from the model
        for block in response.content:
            if block.type == "text":
                click.echo(f"\n[AGENT] {block.text}")

        # If no tool calls, the model is done talking
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_use_blocks:
            self.finished = True
            return

        # Append full response.content (preserves compaction blocks)
        self.messages.append({"role": "assistant", "content": response.content})

        # Execute each tool call
        tool_results = []
        for tool_block in tool_use_blocks:
            result = self._execute_tool(tool_block.name, tool_block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_block.id,
                "content": json.dumps(result) if isinstance(result, dict) else str(result),
            })

        self.messages.append({"role": "user", "content": tool_results})

    def _execute_tool(self, name: str, inputs: dict) -> dict | str:
        """Execute a tool call with validation."""
        if name == "set_hyperparameters":
            return self._tool_set_hyperparameters(inputs)
        elif name == "run_experiment":
            return self._tool_run_experiment()
        elif name == "read_results":
            return self._tool_read_results(inputs)
        elif name == "finish":
            return self._tool_finish(inputs)
        else:
            return {"error": f"Unknown tool: {name}"}

    def _tool_set_hyperparameters(self, inputs: dict) -> dict:
        """Validate and store hyperparameters for the next experiment."""
        if self.guard_triggered:
            return {"error": "Guard has stopped the loop. You must call finish() now.", "accepted": False}

        errors = validate_hyperparameters(inputs, self.config)
        if errors:
            click.echo(f"\n[GUARD] Rejected hyperparameters: {errors}")
            return {"error": f"Validation failed: {errors}", "accepted": False}

        self.current_hp = Hyperparameters(**inputs)
        self._write_train_py()

        click.echo(f"\n[HP SET] model={inputs['base_model']}, lr={inputs['learning_rate']}, "
                    f"bs={inputs['batch_size']}, epochs={inputs['num_epochs']}, "
                    f"seq_len={inputs['max_seq_length']}")
        click.echo(f"[HYPOTHESIS] {inputs['hypothesis']}")

        return {"accepted": True, "hyperparameters": inputs}

    def _tool_run_experiment(self) -> dict:
        """Execute train.py and return results."""
        if self.guard_triggered:
            return {"error": "Guard has stopped the loop. You must call finish() now."}

        if self.current_hp is None:
            return {"error": "No hyperparameters set. Call set_hyperparameters first."}

        self.run_count += 1
        click.echo(f"\n{'='*60}")
        click.echo(f"[RUN {self.run_count}] Starting experiment...")
        click.echo(f"{'='*60}")

        # Execute training in the current terminal (output visible in real-time)
        try:
            result = subprocess.run(
                ["python", "train.py"],
                cwd=self.workspace,
                timeout=self.config.time_budget_seconds + 300,  # extra 5 min buffer
            )
        except subprocess.TimeoutExpired:
            click.echo("[TIMEOUT] Training exceeded time budget")
            return {"error": "Training timed out", "run": self.run_count}

        if result.returncode != 0:
            click.echo("[CRASH] Training failed (see output above)")
            return {
                "error": "Training crashed",
                "run": self.run_count,
            }

        # Read last_result.json
        result_path = self.workspace / "last_result.json"
        if not result_path.exists():
            return {"error": "No last_result.json produced", "run": self.run_count}

        metrics = json.loads(result_path.read_text())
        f1 = metrics.get("f1", 0.0)

        # Check deployment constraints
        violations = []
        if self.config.max_model_size_mb and metrics.get("size_mb", 0) > self.config.max_model_size_mb:
            violations.append(f"size {metrics['size_mb']:.0f}MB > limit {self.config.max_model_size_mb}MB")
        if self.config.max_inference_ms and metrics.get("inference_ms", 0) > self.config.max_inference_ms:
            violations.append(f"latency {metrics['inference_ms']:.1f}ms > limit {self.config.max_inference_ms}ms")

        metrics["constraint_violations"] = violations
        metrics["run"] = self.run_count
        metrics["hypothesis"] = self.current_hp.hypothesis

        # Track best
        improved = f1 > self.best_f1
        if improved and not violations:
            self.best_f1 = f1
            self.best_model = self.current_hp.base_model

        # Log to results.tsv
        self._log_result(metrics, improved)

        # Git commit
        self._git_commit(f"experiment {self.run_count}: {self.current_hp.hypothesis[:60]}")

        click.echo(f"[RESULT] F1={f1:.4f} | precision={metrics.get('precision', 0):.4f} | "
                    f"recall={metrics.get('recall', 0):.4f}")
        click.echo(f"[RESULT] inference={metrics.get('inference_ms', 0):.1f}ms | "
                    f"size={metrics.get('size_mb', 0):.0f}MB | "
                    f"params={metrics.get('params_M', 0):.1f}M")
        if violations:
            click.echo(f"[CONSTRAINT] Violations: {violations}")
        if improved:
            click.echo(f"[IMPROVED] New best F1: {self.best_f1:.4f}")

        # Check guard (safeguards: max runs, no improvement plateau, target metric)
        guard_stop, guard_reason = self._check_guard()
        if guard_stop:
            self.guard_triggered = True
            click.echo(f"\n[GUARD] {guard_reason}")
            click.echo("[GUARD] Stopping — call finish() to end the loop.")
            metrics["guard_stop"] = True
            metrics["guard_reason"] = guard_reason

        return metrics

    def _tool_read_results(self, inputs: dict) -> str:
        """Read results.tsv contents."""
        results_path = self.workspace / "results.tsv"
        if not results_path.exists():
            return "No results yet."

        lines = results_path.read_text().strip().split("\n")
        last_n = inputs.get("last_n", 0)
        if last_n > 0 and len(lines) > last_n + 1:
            # Keep header + last N lines
            lines = [lines[0]] + lines[-(last_n):]

        return "\n".join(lines)

    def _tool_finish(self, inputs: dict) -> dict:
        """End the research loop."""
        self.finished = True

        click.echo(f"\n{'='*60}")
        click.echo("DOCCLOAK MODEL SELECTION")
        click.echo(f"{'='*60}")
        click.echo(f"Selected: {inputs['selected_model']}")
        click.echo(f"F1:       {inputs['f1']:.4f}")
        click.echo(f"Rationale: {inputs['rationale']}")
        click.echo(f"\nDeployment: copy {self.workspace}/best_model/ to DocCloak.Cli model directory")
        click.echo(f"{'='*60}")

        # Log final row
        results_path = self.workspace / "results.tsv"
        with open(results_path, "a") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([
                self.run_count, "FINAL", "final", inputs["selected_model"],
                inputs["f1"], "", "", "", "", "", "",
                inputs["f1"], "", inputs["rationale"][:100],
            ])

        return {"status": "complete", "selected_model": inputs["selected_model"]}

    def _setup_workspace(self) -> None:
        """Generate workspace files from templates."""
        from .scaffold.generator import generate_experiment

        self.workspace.mkdir(parents=True, exist_ok=True)
        generate_experiment(self.config, self.workspace)

        # Git init if needed
        if not (self.workspace / ".git").exists():
            subprocess.run(["git", "init"], cwd=self.workspace, capture_output=True)
            subprocess.run(["git", "add", "."], cwd=self.workspace, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "doccloak-finetune: workspace initialized"],
                cwd=self.workspace, capture_output=True,
            )

    def _init_results_tsv(self) -> None:
        """Initialize results.tsv with the proper header."""
        results_path = self.workspace / "results.tsv"
        if not results_path.exists():
            with open(results_path, "w") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(RESULTS_HEADER)

    def _write_train_py(self) -> None:
        """Write current hyperparameters into train.py."""
        hp = self.current_hp
        train_path = self.workspace / "train.py"
        content = train_path.read_text()

        # Only replace constant definitions at the start of a line (not references in code)
        replacements = {
            r'^(BASE_MODEL\s*=\s*).*': f'BASE_MODEL = "{hp.base_model}"',
            r'^(LEARNING_RATE\s*=\s*).*': f'LEARNING_RATE = {hp.learning_rate}',
            r'^(BATCH_SIZE\s*=\s*).*': f'BATCH_SIZE = {hp.batch_size}',
            r'^(NUM_EPOCHS\s*=\s*).*': f'NUM_EPOCHS = {hp.num_epochs}',
            r'^(WARMUP_RATIO\s*=\s*).*': f'WARMUP_RATIO = {hp.warmup_ratio}',
            r'^(WEIGHT_DECAY\s*=\s*).*': f'WEIGHT_DECAY = {hp.weight_decay}',
            r'^(MAX_SEQ_LENGTH\s*=\s*).*': f'MAX_SEQ_LENGTH = {hp.max_seq_length}',
            r'^(FREEZE_LAYERS\s*=\s*).*': f'FREEZE_LAYERS = {hp.freeze_layers}',
            r'^(GRADIENT_ACCUMULATION_STEPS\s*=\s*).*': f'GRADIENT_ACCUMULATION_STEPS = {hp.gradient_accumulation_steps}',
            r'^(LR_SCHEDULER\s*=\s*).*': f'LR_SCHEDULER = "{hp.lr_scheduler}"',
        }

        for pattern, replacement in replacements.items():
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

        train_path.write_text(content)

    def _log_result(self, metrics: dict, improved: bool) -> None:
        """Append experiment result to results.tsv."""
        results_path = self.workspace / "results.tsv"
        phase = "sweep" if self.run_count <= len(self.config.candidate_models) else "tune"
        notes = self.current_hp.hypothesis[:80] if self.current_hp else ""
        if improved:
            notes += " [IMPROVED]"

        with open(results_path, "a") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([
                self.run_count,
                "",  # tag
                phase,
                self.current_hp.base_model if self.current_hp else "",
                f"{metrics.get('f1', 0):.4f}",
                f"{metrics.get('precision', 0):.4f}",
                f"{metrics.get('recall', 0):.4f}",
                f"{metrics.get('loss', 0):.4f}",
                f"{metrics.get('params_M', 0):.1f}",
                f"{metrics.get('size_mb', 0):.0f}",
                f"{metrics.get('inference_ms', 0):.1f}",
                f"{metrics.get('f1', 0):.4f}",  # value column (for guard.py compat)
                f"{metrics.get('duration_s', 0):.0f}",
                notes,
            ])

    def _check_guard(self) -> tuple[bool, str]:
        """Run guard.py to check if safeguards should stop the loop."""
        guard_path = self.workspace / "guard.py"
        if not guard_path.exists():
            return False, ""

        try:
            result = subprocess.run(
                ["python", "guard.py"],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=10,
            )
            reason = result.stdout.strip()
            return result.returncode != 0, reason
        except Exception:
            return False, ""

    def _git_commit(self, message: str) -> None:
        """Commit current workspace state."""
        subprocess.run(["git", "add", "-A"], cwd=self.workspace, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", f"doccloak-finetune: {message}"],
            cwd=self.workspace, capture_output=True,
        )
