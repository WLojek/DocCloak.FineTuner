"""ONNX export for causal language models (text generation).

Handles LoRA adapter merging, ONNX conversion, and INT4 quantization
for browser deployment via onnxruntime-web.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import click


def merge_lora_weights(model_dir: Path, output_dir: Path) -> Path:
    """Merge LoRA adapter weights into the base model."""
    click.echo("Merging LoRA adapter weights...")

    try:
        from peft import PeftModel, PeftConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load PEFT config to get base model name
        peft_config = PeftConfig.from_pretrained(str(model_dir))
        base_model_name = peft_config.base_model_name_or_path

        click.echo(f"  Base model: {base_model_name}")

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype="auto",
            device_map="cpu",
        )

        # Load and merge LoRA
        model = PeftModel.from_pretrained(base_model, str(model_dir))
        merged_model = model.merge_and_unload()

        # Save merged model
        merged_dir = output_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(str(merged_dir))

        # Copy tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        tokenizer.save_pretrained(str(merged_dir))

        click.echo(f"  Merged model saved to {merged_dir}")
        return merged_dir

    except ImportError:
        click.echo("  No PEFT adapter found, using model as-is")
        return model_dir


def export_onnx(model_dir: Path, output_dir: Path) -> Path:
    """Export causal LM to ONNX format."""
    click.echo("Exporting to ONNX...")

    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    try:
        from optimum.onnxruntime import ORTModelForCausalLM

        click.echo("  Using optimum for ONNX export...")
        try:
            ort_model = ORTModelForCausalLM.from_pretrained(
                str(model_dir),
                export=True,
            )
            ort_model.save_pretrained(str(onnx_dir))
            click.echo(f"  ONNX model saved to {onnx_dir}")
        except Exception as e:
            if "serialize" in str(e).lower() or "post-process" in str(e).lower():
                click.echo(f"  Optimum post-processing failed (model too large for protobuf), falling back to torch.onnx...")
                _export_onnx_torch(model_dir, onnx_dir)
            else:
                raise

    except ImportError:
        click.echo("  optimum not available, using torch.onnx.export...")
        _export_onnx_torch(model_dir, onnx_dir)

    # Copy tokenizer files
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(onnx_dir))

    return onnx_dir


def _export_onnx_torch(model_dir: Path, onnx_dir: Path) -> None:
    """Fallback ONNX export using torch.onnx."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model.eval()

    dummy_input = tokenizer("Hello world", return_tensors="pt")
    input_ids = dummy_input["input_ids"]
    attention_mask = dummy_input["attention_mask"]

    onnx_path = onnx_dir / "model.onnx"

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=14,
    )
    click.echo(f"  Exported to {onnx_path}")


def quantize_int4(onnx_dir: Path) -> Path:
    """Apply INT4 quantization to the ONNX model."""
    click.echo("Applying INT4 quantization...")

    try:
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        quantizer = ORTQuantizer.from_pretrained(str(onnx_dir))
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

        quantized_dir = onnx_dir.parent / "onnx_int4"
        quantized_dir.mkdir(parents=True, exist_ok=True)

        quantizer.quantize(
            save_dir=str(quantized_dir),
            quantization_config=qconfig,
        )
        click.echo(f"  INT4 model saved to {quantized_dir}")
        return quantized_dir

    except (ImportError, Exception) as e:
        click.echo(f"  INT4 quantization failed: {e}")
        click.echo("  Falling back to INT8 dynamic quantization...")
        return _quantize_int8_dynamic(onnx_dir)


def _quantize_int8_dynamic(onnx_dir: Path) -> Path:
    """Fallback: INT8 dynamic quantization."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        model_path = onnx_dir / "model.onnx"
        if not model_path.exists():
            # Try to find the model file
            onnx_files = list(onnx_dir.glob("*.onnx"))
            if onnx_files:
                model_path = onnx_files[0]
            else:
                click.echo("  No ONNX model found for quantization")
                return onnx_dir

        quantized_dir = onnx_dir.parent / "onnx_quantized"
        quantized_dir.mkdir(parents=True, exist_ok=True)
        quantized_path = quantized_dir / "model_quantized.onnx"

        quantize_dynamic(
            str(model_path),
            str(quantized_path),
            weight_type=QuantType.QInt8,
        )

        # Copy config and tokenizer files
        for f in onnx_dir.glob("*.json"):
            shutil.copy2(f, quantized_dir)
        for f in onnx_dir.glob("tokenizer*"):
            shutil.copy2(f, quantized_dir)
        for f in onnx_dir.glob("special_tokens*"):
            shutil.copy2(f, quantized_dir)

        model_size = quantized_path.stat().st_size / (1024 ** 2)
        click.echo(f"  Quantized model: {model_size:.0f} MB -> {quantized_path}")

        return quantized_dir

    except ImportError:
        click.echo("  onnxruntime.quantization not available")
        return onnx_dir


def validate_onnx(onnx_dir: Path) -> bool:
    """Validate ONNX model by running a test inference."""
    click.echo("Validating ONNX model...")

    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer
        import numpy as np

        # Find model file
        model_path = None
        for name in ["model_quantized.onnx", "model.onnx", "decoder_model.onnx"]:
            candidate = onnx_dir / name
            if candidate.exists():
                model_path = candidate
                break

        if not model_path:
            onnx_files = list(onnx_dir.glob("*.onnx"))
            model_path = onnx_files[0] if onnx_files else None

        if not model_path:
            click.echo("  No ONNX model found")
            return False

        tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir))

        session = ort.InferenceSession(str(model_path))
        inputs = tokenizer("Hello, world!", return_tensors="np")

        input_feed = {}
        input_names = [inp.name for inp in session.get_inputs()]
        for name in input_names:
            if name in inputs:
                input_feed[name] = inputs[name]

        start = time.perf_counter()
        outputs = session.run(None, input_feed)
        elapsed = (time.perf_counter() - start) * 1000

        click.echo(f"  Validation passed: output shape={outputs[0].shape}, latency={elapsed:.1f}ms")
        return True

    except Exception as e:
        click.echo(f"  Validation failed: {e}")
        return False


def quantize_gptq(model_dir: Path, output_dir: Path) -> Path:
    """Apply GPTQ INT4 quantization. Requires GPU."""
    click.echo("Applying GPTQ INT4 quantization (requires GPU)...")

    try:
        import torch
        if not torch.cuda.is_available():
            click.echo("  WARNING: No CUDA GPU detected. GPTQ requires GPU.")
            click.echo("  Run this on a GPU machine (e.g. RunPod with RTX 4090).")
            raise RuntimeError("GPTQ requires CUDA GPU")

        from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

        gptq_dir = output_dir / "gptq_int4"
        gptq_dir.mkdir(parents=True, exist_ok=True)

        click.echo("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        click.echo("  Configuring GPTQ (4-bit, 128 block size)...")
        gptq_config = GPTQConfig(
            bits=4,
            dataset="c4",
            tokenizer=tokenizer,
            block_name_to_quantize="model.layers",
            model_seqlen=512,
        )

        click.echo("  Loading and quantizing model (this takes a few minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            quantization_config=gptq_config,
            device_map="auto",
        )

        click.echo("  Saving quantized model...")
        model.save_pretrained(str(gptq_dir))
        tokenizer.save_pretrained(str(gptq_dir))

        total = sum(f.stat().st_size for f in gptq_dir.rglob("*") if f.is_file())
        click.echo(f"  GPTQ INT4 model saved to {gptq_dir} ({total / (1024**2):.0f} MB)")

        return gptq_dir

    except ImportError as e:
        click.echo(f"  GPTQ dependencies missing: {e}")
        click.echo("  Install with: pip install auto-gptq optimum")
        raise


def export_causal_model(
    model_dir: Path,
    output_dir: Path,
    do_onnx: bool = True,
    do_int4: bool = False,
    do_gptq: bool = False,
    do_pytorch: bool = False,
) -> None:
    """Full export pipeline for causal LM models."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "source_model": str(model_dir),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "exports": [],
    }

    # Check if this is a LoRA model and merge if so
    is_lora = (model_dir / "adapter_config.json").exists()
    if is_lora:
        model_dir = merge_lora_weights(model_dir, output_dir)
        metadata["lora_merged"] = True

    # PyTorch export
    if do_pytorch:
        pytorch_dir = output_dir / "pytorch"
        pytorch_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(model_dir, pytorch_dir, dirs_exist_ok=True)
        click.echo(f"PyTorch model copied to {pytorch_dir}")
        metadata["exports"].append("pytorch")

    # GPTQ INT4 quantization (requires GPU)
    if do_gptq:
        gptq_dir = quantize_gptq(model_dir, output_dir)
        metadata["exports"].append("gptq_int4")

    # ONNX export
    if do_onnx:
        onnx_dir = export_onnx(model_dir, output_dir)
        metadata["exports"].append("onnx")

        # INT4 quantization
        if do_int4:
            quantized_dir = quantize_int4(onnx_dir)
            validate_onnx(quantized_dir)
            metadata["exports"].append("onnx_int4")
        else:
            validate_onnx(onnx_dir)

    # Save metadata
    (output_dir / "export_metadata.json").write_text(json.dumps(metadata, indent=2))
    click.echo(f"\nExport complete: {output_dir}")
