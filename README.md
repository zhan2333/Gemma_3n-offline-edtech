# Gemma_3n-offline-edtech

让 **Gemma 3n** 在 0 网络现场当老师，Edge TPU 超充，教育平权。

---

## ⚡ Quick Start

```bash
# 0. 克隆仓库
git clone git@github.com:zhan2333/Gemma_3n-offline-edtech.git
cd Gemma_3n-offline-edtech   # ⏰ <5 s

# 1. 创建并激活环境（≈2 min）
conda env create -f environment.yml
conda activate gemma-demo

# 2. 跑 Demo（<1 s）
python src/infer/cli.py
# → 👋 Offline Gemma 3n says hi!
