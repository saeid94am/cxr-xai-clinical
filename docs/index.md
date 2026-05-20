# CXR-XAI-Clinical

**Multi-label chest X-ray classification with quantitative explainability.**

This project implements DenseNet-121 and ViT-Base/16 on NIH ChestX-ray14,
with four XAI methods (Grad-CAM++, HiResCAM, Integrated Gradients, Attention Rollout)
evaluated across six quantitative faithfulness and stability metrics.

## Framework

- **PyTorch + timm** for 2D classification and XAI — chosen over MONAI because
  2D classification does not need MONAI's volumetric pipeline, and full gradient
  access is required for Grad-CAM++ and HiResCAM.
- **grad-cam** for CAM methods, **captum** for Integrated Gradients,
  **quantus** for ROAD faithfulness metric.

## Quick start

```bash
# 1. Download data
bash scripts/download_data.sh

# 2. Train
python scripts/train.py --config configs/train.yaml --model densenet121

# 3. Evaluate
bash scripts/run_experiments.sh --model densenet121
```

See [Getting Started](getting-started/installation.md) for full setup instructions.
