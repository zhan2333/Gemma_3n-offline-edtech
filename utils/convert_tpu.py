#!/usr/bin/env python3
"""
convert_tpu.py
===============
One-stop script that takes a **Gemma 3n HF directory + (optional) LoRA** and produces:

1. *merged*    : `gemma3n-merged/` → HF fp32 (or bfloat) weights (≈7 GB)
2. *ONNX*      : `model_fp16.onnx`   (≈3.5 GB)
3. *INT8 TFLite* : `gemma3n_int8.tflite` (size will still be huge, see NOTE)
4. *(Optional)* *Edge-TPU* compiled model : `gemma3n_int8_edgetpu.tflite`

The script is **idempotent** – it skips steps whose outputs already exist.
If an Edge-TPU compiler is *not* found on PATH, step ❹ is skipped with a hint.

Run:
    python utils/convert_tpu.py \
        --base /path/to/gemma3n-hf \
        --lora models/skill_math_3n.lora \
        --out models/tpu_build

Dependencies (see environment.yml):
  • peft, torch, transformers, optimum[onnx]
  • onnx, tf2onnx, tensorflow (>=2.15), tflite-support
  • Edge-TPU compiler ≥16 (optional)

NOTE:
-----
A full 6.9B-param model will *not* fit Edge-TPU 10 MB limit even after int8.
This script is mainly for **pipeline completeness** & benchmarking.
For an actual deployable TPU model you should:
  • switch to gemma3n:e2b, **or**
  • truncate layers / use LoRA-adapter on last N blocks.
"""

from pathlib import Path
import argparse, subprocess, os, sys, json, shutil

# ----------------------- helpers ------------------------------------------------

def run(cmd: list[str]):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)

# ----------------------- pipeline steps ----------------------------------------

def merge_lora(base_dir: Path, lora_path: Path, merged_out: Path):
    if merged_out.exists():
        print("✓ merged weights already exist, skip")
        return
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    import torch

    model = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype="auto")
    if lora_path and lora_path.exists():
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
    model.save_pretrained(merged_out)
    print("✅ merged →", merged_out)


def export_onnx(merged_dir: Path, onnx_out: Path):
    if onnx_out.exists():
        print("✓ ONNX already exists, skip")
        return
    run([
        "python", "-m", "optimum.exporters.onnx", "--model", str(merged_dir),
        "--quantization", "fp16", "--task", "causal-lm", str(onnx_out.parent)
    ])


def convert_tflite(onnx_model: Path, tflite_out: Path):
    if tflite_out.exists():
        print("✓ TFLite already exists, skip")
        return
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf
    import numpy as np

    tmp_saved = tflite_out.parent / "tf_saved"
    tmp_saved.mkdir(exist_ok=True)

    # ONNX → TF
    tf_rep = prepare(onnx.load(onnx_model), device="CPU")
    tf_rep.export_graph(tmp_saved)

    # TF → INT8 TFLite
    def rep_data():
        for _ in range(50):
            yield [np.random.randint(0, 32000, size=(1, 128), dtype=np.int32)]

    conv = tf.lite.TFLiteConverter.from_saved_model(tmp_saved)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_data
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    tfl = conv.convert()
    tflite_out.write_bytes(tfl)
    shutil.rmtree(tmp_saved)
    print("✅ INT8 TFLite →", tflite_out)


def compile_edgetpu(tflite_int8: Path, edgetpu_out: Path):
    if edgetpu_out.exists():
        print("✓ Edge-TPU model already exists, skip")
        return
    compiler = shutil.which("edgetpu_compiler")
    if not compiler:
        print("⚠️  edgetpu_compiler not found – skipping TPU compilation.")
        return
    run([compiler, "-o", str(edgetpu_out.parent), str(tflite_int8)])

# ----------------------- CLI ----------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="HF directory of Gemma 3n")
    ap.add_argument("--lora", help="Path to LoRA .lora file (optional)")
    ap.add_argument("--out", default="tpu_build", help="Output work dir")
    args = ap.parse_args()

    work = Path(args.out).resolve()
    work.mkdir(exist_ok=True)

    merged_dir  = work / "merged"
    onnx_dir    = work / "onnx"
    onnx_model  = onnx_dir / "model_fp16.onnx"
    tflite_int8 = work / "gemma3n_int8.tflite"
    tpu_model   = work / "gemma3n_int8_edgetpu.tflite"

    merge_lora(Path(args.base), Path(args.lora) if args.lora else None, merged_dir)
    export_onnx(merged_dir, onnx_model)
    convert_tflite(onnx_model, tflite_int8)
    compile_edgetpu(tflite_int8, tpu_model)

if __name__ == "__main__":
    main()

