"""Model export and quantization for DocCloak.FineTuner.

Supports exporting trained models to:
- PyTorch (copy as-is)
- ONNX (for cross-platform / browser deployment)

Quantization options:
- none: keep FP32 weights
- dynamic: INT8 dynamic quantization (fastest, no calibration needed)
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import click
import torch


def _make_calibration_reader(dataset_path: Path, tokenizer, num_samples: int = 100):
    """Build a calibration data reader for static quantization.

    Must be called after onnxruntime is confirmed importable, since
    CalibrationDataReader lives there.
    """
    import numpy as np
    from datasets import load_from_disk
    from onnxruntime.quantization import CalibrationDataReader

    class _Reader(CalibrationDataReader):
        def __init__(self):
            ds = load_from_disk(str(dataset_path))
            split = ds.get("validation", ds.get("test", ds["train"]))

            self.samples = []
            for i, example in enumerate(split):
                if i >= num_samples:
                    break
                text = " ".join(example["tokens"])
                inputs = tokenizer(
                    text, return_tensors="np", padding="max_length",
                    truncation=True, max_length=128,
                )
                self.samples.append({
                    "input_ids": inputs["input_ids"].astype(np.int64),
                    "attention_mask": inputs["attention_mask"].astype(np.int64),
                })
            self.index = 0

        def get_next(self):
            if self.index >= len(self.samples):
                return None
            sample = self.samples[self.index]
            self.index += 1
            return sample

    return _Reader()



def _fix_tokenizer_config(model_dir: Path) -> None:
    """Fix tokenizer_config.json if extra_special_tokens is a list instead of dict.

    Some transformers versions save this field as a list, but newer versions
    expect a dict. This patches the file in-place to prevent AttributeError.
    """
    config_path = model_dir / "tokenizer_config.json"
    if not config_path.exists():
        return

    with open(config_path) as f:
        config = json.load(f)

    changed = False
    if "extra_special_tokens" in config and isinstance(config["extra_special_tokens"], list):
        config["extra_special_tokens"] = {}
        changed = True

    if changed:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)


def _validate_model_dir(model_dir: Path) -> None:
    """Check that the model directory contains required files."""
    required = ["config.json"]
    has_weights = (
        (model_dir / "pytorch_model.bin").exists()
        or (model_dir / "model.safetensors").exists()
    )
    if not has_weights:
        raise click.ClickException(
            f"No model weights found in {model_dir}. "
            "Expected pytorch_model.bin or model.safetensors."
        )
    for f in required:
        if not (model_dir / f).exists():
            raise click.ClickException(f"Missing required file: {model_dir / f}")


def _copy_tokenizer_files(src: Path, dst: Path) -> None:
    """Copy tokenizer and config files to output directory."""
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "added_tokens.json",
        "config.json",
    ]
    for f in tokenizer_files:
        src_file = src / f
        if src_file.exists():
            shutil.copy2(src_file, dst / f)


def _dir_size_mb(path: Path) -> float:
    """Calculate total size of a directory in MB."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def export_pytorch(model_dir: Path, output_dir: Path) -> Path:
    """Copy PyTorch model to output directory."""
    pytorch_dir = output_dir / "pytorch"
    pytorch_dir.mkdir(parents=True, exist_ok=True)

    for f in model_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, pytorch_dir / f.name)

    size = _dir_size_mb(pytorch_dir)
    click.echo(f"  PyTorch model saved to {pytorch_dir} ({size:.1f} MB)")
    return pytorch_dir


def export_onnx(
    model_dir: Path,
    output_dir: Path,
    quantize: str = "none",
    calibration_data: Path | None = None,
) -> Path:
    """Export model to ONNX format with optional quantization.

    Args:
        model_dir: Path to trained HuggingFace model directory.
        output_dir: Root output directory.
        quantize: "none", "dynamic", or "static".
        calibration_data: Path to HF dataset for static quantization calibration.
    """
    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError:
        raise click.ClickException("transformers is required for ONNX export.")

    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    click.echo("  Loading model...")
    model = AutoModelForTokenClassification.from_pretrained(model_dir)

    # Fix tokenizer config if needed — some transformers versions save
    # extra_special_tokens as a list, but newer versions expect a dict
    _fix_tokenizer_config(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()
    model.cpu()

    click.echo("  Exporting to ONNX...")
    onnx_path = onnx_dir / "model.onnx"

    # Use optimum if available (handles newer transformers SDPA attention correctly)
    try:
        from optimum.onnxruntime import ORTModelForTokenClassification

        click.echo("  Using optimum for ONNX export...")
        ort_model = ORTModelForTokenClassification.from_pretrained(
            model_dir, export=True
        )
        ort_model.save_pretrained(onnx_dir)

        # optimum saves as model.onnx in the output dir
        if not onnx_path.exists():
            # Check for alternative name
            for candidate in onnx_dir.glob("*.onnx"):
                candidate.rename(onnx_path)
                break

    except ImportError:
        click.echo("  optimum not found, falling back to torch.onnx.export...")
        # Disable SDPA to avoid tracing issues with newer transformers
        model.config.attn_implementation = "eager"
        model = AutoModelForTokenClassification.from_pretrained(
            model_dir, attn_implementation="eager"
        )
        model.eval()
        model.cpu()

        dummy_input = tokenizer(
            "Przykładowy tekst do eksportu modelu.",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )

        torch.onnx.export(
            model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=14,
        )

    onnx_size = onnx_path.stat().st_size / (1024 * 1024)
    click.echo(f"  ONNX model exported ({onnx_size:.1f} MB)")

    if quantize in ("dynamic", "static"):
        try:
            from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, CalibrationDataReader
        except ImportError:
            raise click.ClickException(
                "onnxruntime is required for quantization. "
                "Install with: pip install onnxruntime"
            )

        quantized_path = onnx_dir / "model_quantized.onnx"

        if quantize == "dynamic":
            click.echo("  Applying INT8 dynamic quantization...")
            quantize_dynamic(
                str(onnx_path),
                str(quantized_path),
                weight_type=QuantType.QUInt8,
            )
        else:
            click.echo("  Applying INT8 static quantization (calibrating)...")
            if calibration_data is None:
                raise click.ClickException(
                    "Static quantization requires --calibration-data"
                )
            calibration_reader = _make_calibration_reader(
                calibration_data, tokenizer, num_samples=100
            )
            quantize_static(
                str(onnx_path),
                str(quantized_path),
                calibration_reader,
                weight_type=QuantType.QInt8,
            )

        quantized_size = quantized_path.stat().st_size / (1024 * 1024)
        reduction = (1 - quantized_size / onnx_size) * 100
        click.echo(
            f"  Quantized model saved ({quantized_size:.1f} MB, "
            f"{reduction:.0f}% reduction)"
        )

    _copy_tokenizer_files(model_dir, onnx_dir)

    # Save export metadata
    label_map = {}
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            model_config = json.load(f)
        label_map = model_config.get("id2label", {})

    metadata = {
        "source_model": str(model_dir),
        "format": "onnx",
        "opset_version": 14,
        "quantization": quantize,
        "onnx_size_mb": round(onnx_size, 1),
        "id2label": label_map,
    }
    if quantize != "none":
        metadata["quantized_size_mb"] = round(quantized_size, 1)

    with open(onnx_dir / "export_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    total_size = _dir_size_mb(onnx_dir)
    click.echo(f"  ONNX export complete: {onnx_dir} ({total_size:.1f} MB total)")
    return onnx_dir


def validate_onnx(onnx_dir: Path) -> None:
    """Run a quick validation on the exported ONNX model."""
    try:
        import onnxruntime as ort
    except ImportError:
        click.echo("  Skipping ONNX validation (onnxruntime not installed)")
        return

    # Find the model file (prefer quantized if available)
    quantized_path = onnx_dir / "model_quantized.onnx"
    onnx_path = onnx_dir / "model.onnx"
    model_path = quantized_path if quantized_path.exists() else onnx_path

    if not model_path.exists():
        click.echo("  Skipping validation (no ONNX model found)")
        return

    click.echo(f"  Validating {model_path.name}...")

    try:
        from transformers import AutoTokenizer
    except ImportError:
        click.echo("  Skipping validation (transformers not installed)")
        return

    tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir))
    session = ort.InferenceSession(str(model_path))

    test_texts = [
        "Jan Kowalski mieszka w Warszawie.",
        "PESEL 92030512345, NIP 1234567890.",
        "Faktura VAT nr 2024/01/001.",
    ]

    import numpy as np

    # Check which inputs the model expects
    model_inputs = {inp.name for inp in session.get_inputs()}

    for text in test_texts:
        inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
        feed = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        if "token_type_ids" in model_inputs:
            feed["token_type_ids"] = inputs.get(
                "token_type_ids", np.zeros_like(inputs["input_ids"])
            ).astype(np.int64)
        outputs = session.run(None, feed)
        logits = outputs[0]
        predictions = np.argmax(logits, axis=-1)
        non_o = int(np.sum(predictions != 0))
        click.echo(f"    \"{text[:50]}\" -> {non_o} entities detected")

    # Benchmark speed
    inputs = tokenizer(test_texts[0], return_tensors="np", padding=True, truncation=True)
    feed = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }
    if "token_type_ids" in model_inputs:
        feed["token_type_ids"] = inputs.get(
            "token_type_ids", np.zeros_like(inputs["input_ids"])
        ).astype(np.int64)

    # Warmup
    for _ in range(5):
        session.run(None, feed)

    # Benchmark
    times = []
    for _ in range(50):
        start = time.perf_counter()
        session.run(None, feed)
        times.append((time.perf_counter() - start) * 1000)

    avg_ms = sum(times) / len(times)
    click.echo(f"  ONNX inference: {avg_ms:.1f} ms/sample (avg over 50 runs)")
    click.echo("  Validation passed!")
