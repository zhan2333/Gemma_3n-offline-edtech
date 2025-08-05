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
