# Gemma_3n-offline-edtech

让 **Gemma 3n** 在 0 网络现场当老师，Edge TPU 超充，教育平权。

---

## ⚡ Quick Start (100 % 离线)

```bash
# 0. 克隆仓库
git clone git@github.com:zhan2333/Gemma_3n-offline-edtech.git
cd Gemma_3n-offline-edtech          # ⏱ <5 s

# 1. 安装/激活环境（≈2 min）
conda env create -f environment.yml
conda activate gemma3n

# 2. 拉基础模型（一次 7 GB，可断网复用）
ollama pull gemma3n

# 3. 获取 LoRA 权重（仅几 MB）
mkdir -p models
wget -P models/ \
  https://huggingface.co/zhan23333/gemma3n-loras/resolve/main/skill_math_3n.lora

# 4. 离线推理
python src/offline_infer.py --prompt "你好，世界！"
# → 👋 Offline Gemma 3n 回应
```

---

## 📦 目录结构

```
Gemma_3n-offline-edtech/
├─ gemma3n-hf/              # ← export 脚本产物（.gitignore 忽略）
├─ models/                  # ← LoRA 权重放这里
│   └─ skill_math_3n.lora
├─ utils/
│   ├─ export_gemma3n.py    # base → HF 目录
│   └─ convert_tpu.py       # HF(+LoRA) → ONNX→TFLite→Edge-TPU
├─ src/
│   └─ offline_infer.py     # 离线 CLI / API
├─ data/                    # 训练样例 (jsonl)
├─ environment.yml
└─ README.md
```

---

## 🚀 功能概览

| 模块 | 说明 |
| --- | --- |
| `utils/export_gemma3n.py` | 调 `ollama.convert_to_hf()`，把 **gemma3n** 权重导成标准 Hugging Face 目录 |
| `train_lora.py` | 使用 **UnsLoTH** 极简微调，产出 `.lora` 文件 |
| `utils/convert_tpu.py` | 一键管线：merge LoRA → ONNX(fp16) → TFLite(int8) → *(可选)* Edge-TPU `.tflite` |
| `src/offline_infer.py` | 离线 CLI/REST，自动检测 `models/*.lora` |
| `kaggle_demo.ipynb` | Kaggle GPU 上完整复现：解压 HF → 微调 → 量化 |

---

## 🏁 里程碑

### 1️⃣ LoRA 快速微调

```bash
python train_lora.py \
  --base gemma3n-hf \
  --data data/math.jsonl \
  --epochs 1 --lr 1e-4
# 输出 models/skill_math_3n.lora
```

### 2️⃣ Edge-TPU 转换 & Benchmark

```bash
# 导出 HF 权重（一次性）
python utils/export_gemma3n.py --model gemma3n --out gemma3n-hf

# 转换 / 量化
python utils/convert_tpu.py \
  --base gemma3n-hf \
  --lora models/skill_math_3n.lora \
  --out  tpu_build
# 无 edgetpu_compiler 时脚本会自动跳过 TPU 编译并提示
```
若无 edgetpu_compiler，脚本自动跳过 TPU 编译并提示。

---

## 📊 结果示例

| Stage        | 文件                             | Size |
|--------------|----------------------------------|------|
| merged fp32  | `merged/pytorch_model.bin`       | 6.9&nbsp;GB |
| ONNX fp16    | `onnx/model_fp16.onnx`           | 3.5&nbsp;GB |
| TFLite int8  | `gemma3n_int8.tflite`            | 1.6&nbsp;GB |
| Edge-TPU     | *(模型超 10 MB，上 TPU 受限)*    | — |

GPU T4 推理 ≈ 71 tok/s；Mac M3 CPU int8 ≈ 7 tok/s。

---

## 📝 Tech Report & Demo

| 链接          | 内容                                   |
|---------------|----------------------------------------|
| **Tech Report** | 架构、微调日志、Edge-TPU 转换记录       |
| **3-min Video** | 断网推理演示 + Edge-TPU 加速画面        |
| **HF Space**    | 在线体验（纯 CPU，可选）               |

---

## 🛠 开发者速查

```bash
# 导出 Hugging Face 权重
python utils/export_gemma3n.py --model gemma3n --out gemma3n-hf

# 一键 ONNX / TFLite / Edge-TPU
python utils/convert_tpu.py --base gemma3n-hf --out tpu_build

# 本地推理
python src/offline_infer.py --prompt "Explain Pythagorean theorem."
```

---

## 📜 License

代码 Apache-2.0；模型权重遵循 Google Gemma License。

*Made with ♥ by zhan2333 — 为弱网教育场景打造的离线 LLM 工具箱。*
