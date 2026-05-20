# cxr-xai-clinical

**Chest X-Ray Classification + Quantitative XAI** — Phase 2 of a clinical AI portfolio.

The central contribution is a systematic, quantitative evaluation of post-hoc explainability methods across CNN and Transformer architectures for multi-label chest pathology detection, demonstrating that **visual plausibility and mathematical faithfulness of saliency maps are dissociated**.

---

## Models

| Model | Role | VRAM (fp16) |
|---|---|---|
| DenseNet-121 (`timm`) | CNN baseline (CheXNet architecture) | ~2.8 GB @ bs=16 |
| ViT-Base/16 (`timm`) | Transformer baseline | ~3.4 GB @ bs=8 |

## Datasets

| Dataset | Purpose |
|---|---|
| NIH ChestX-ray14 | Primary training + evaluation (14 pathologies, 112 K images) |
| CheXpert val set | Cross-dataset generalization (5-pathology eval) |
| RSNA Pneumonia | Domain shift XAI experiment (optional) |

## XAI Methods

| Method | Backbone | Library |
|---|---|---|
| Grad-CAM++ | DenseNet-121 | `pytorch-grad-cam` |
| HiResCAM | DenseNet-121 | `pytorch-grad-cam` |
| Integrated Gradients | ViT-Base/16 | `captum` |
| Attention Rollout | ViT-Base/16 | custom |

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Train DenseNet-121 in debug mode (20 K subset, 2 epochs)
python scripts/train.py --config configs/train.yaml --debug

# Train full run
python scripts/train.py --config configs/train.yaml --model densenet121

# Generate XAI heatmaps
python scripts/generate_xai.py --config configs/xai.yaml --checkpoint results/checkpoints/best_densenet121.pt

# Evaluate XAI metrics
python scripts/evaluate.py --config configs/xai.yaml

# Run Streamlit demo
streamlit run demo/app.py
```

## Docker

```bash
docker-compose up train
```

## Compute strategy

- **Full training**: Kaggle Notebooks (P100/T4, 16 GB VRAM) — data already hosted on Kaggle
- **XAI generation + demo**: Local RTX 3050 (4 GB VRAM, inference only)
- **Debug / hyperparameter search**: Local RTX 3050 on 20 K subsample

## Results structure

```
results/
├── checkpoints/        # best_<model>.pt + last_<model>.pt
├── metrics/
│   ├── auroc_nih14.csv
│   ├── auroc_chexpert.csv
│   ├── xai_comparison_table.csv
│   └── xai_domain_shift.csv
└── figures/            # qualitative heatmap grids
```

## Future extensions

- Tabular metadata (age, sex, view position) as a SHAP branch — deliberately excluded from this phase; justified in Phase2 plan
- Phase 5: ECG + CXR multimodal fusion
