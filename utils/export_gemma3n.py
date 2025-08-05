#!/usr/bin/env python3
"""
Export local Ollama gemma3n weights into HuggingFace format.
Works on Ollama 0.10.1 (no export command).
"""
import json, os, pathlib, shutil, sys, glob

MODEL_NAME = "gemma3n"           # 换成 "gemma3n:e2b" 可省空间
ROOT = pathlib.Path.home() / ".ollama" / "models"

# 1. 找 manifest
manifest = None
for mf in (ROOT / "manifests").glob("*.json"):
    if json.load(open(mf)).get("name") == MODEL_NAME:
        manifest = mf
        break
if manifest is None:
    sys.exit(f"❌ 先执行 `ollama pull {MODEL_NAME}` 再来")

model_sha = manifest.stem
blob_root = ROOT / "blobs"

# 2. 输出目录
out = pathlib.Path("gemma3n-hf")
out.mkdir(exist_ok=True)

# 3. 拷模型 & 配置
shutil.copy(manifest, out / "config.json")
# model.bin 在 blobs/sha256/XX/YY/<sha>/
bin_path = next(blob_root.glob(f"**/{model_sha}/model.bin"))
shutil.copy(bin_path, out / "model.safetensors")

# 4. tokenizer 文件
for tok_name in ("tokenizer.json", "tokenizer_config.json"):
    tok_src = next(blob_root.glob(f"**/{model_sha}/{tok_name}"), None)
    if tok_src:
        shutil.copy(tok_src, out / tok_name)

print(f"✅ 导出完成 → {out.resolve()}")
