"""
CXR-XAI-Clinical — Streamlit Demo

Upload a chest X-ray → get:
  1. Predicted pathologies with confidence scores
  2. Side-by-side XAI heatmap overlays (method depends on loaded model)
  3. Per-image quantitative metric summary

Run:
    streamlit run demo/app.py
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.transforms import val_transforms
from src.models import build_model
from src.training.checkpoint import load_checkpoint
from src.xai import (
    compute_cam_batch,
    compute_integrated_gradients,
    compute_attention_rollout,
)
from src.evaluation.deletion_insertion import compute_deletion_insertion
from src.evaluation.stability_sanity import compute_spearman_stability
from src.data.dataset import NIH14_CLASSES

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CXR-XAI Clinical",
    page_icon="🫁",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = Path("results/checkpoints")
MODEL_OPTIONS = {
    "DenseNet-121": "densenet121",
    "ViT-Base/16":  "vit_base_patch16_224",
}
XAI_METHODS = {
    "densenet121":          ["Grad-CAM++", "HiResCAM"],
    "vit_base_patch16_224": ["Integrated Gradients", "Attention Rollout"],
}
METHOD_KEY = {
    "Grad-CAM++":           "gradcam_plus_plus",
    "HiResCAM":             "hirescam",
    "Integrated Gradients": "integrated_gradients",
    "Attention Rollout":    "attention_rollout",
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Model loading (cached) ────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model weights…")
def load_model(model_name: str, checkpoint_path: str):
    model = build_model(model_name, num_classes=14, pretrained=False).to(DEVICE)
    load_checkpoint(checkpoint_path, model, device=DEVICE)
    model.eval()
    return model


# ── Heatmap overlay helper ────────────────────────────────────────────────────

def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a (H,W) saliency map over an RGB image as a jet-coloured overlay."""
    import cv2

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)

    img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
    blended = (alpha * coloured + (1 - alpha) * img_uint8).astype(np.uint8)
    return blended


def tensor_to_display(tensor: torch.Tensor) -> np.ndarray:
    """Denormalise a (3,H,W) tensor back to [0,1] RGB for display."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = tensor.permute(1, 2, 0).numpy()
    img = img * std + mean
    return img.clip(0, 1)


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("Settings")

model_display = st.sidebar.selectbox("Model", list(MODEL_OPTIONS.keys()))
model_name    = MODEL_OPTIONS[model_display]

ckpt_path = CHECKPOINT_DIR / f"best_{model_name}.pt"
if not ckpt_path.exists():
    st.sidebar.warning(f"No checkpoint found at `{ckpt_path}`.\nTrain the model first.")
    model_loaded = False
else:
    model = load_model(model_name, str(ckpt_path))
    model_loaded = True
    st.sidebar.success(f"Loaded `{ckpt_path.name}`")

available_methods = XAI_METHODS[model_name]
selected_methods  = st.sidebar.multiselect(
    "XAI methods", available_methods, default=available_methods
)

conf_threshold = st.sidebar.slider(
    "Confidence threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05
)

show_metrics = st.sidebar.checkbox("Show quantitative metrics", value=True)

# ── Main ──────────────────────────────────────────────────────────────────────

st.title("Chest X-Ray Pathology Classification + XAI")
st.caption(
    "Upload a frontal CXR. The model predicts 14 NIH pathologies and generates "
    "saliency maps with quantitative faithfulness metrics."
)

uploaded = st.file_uploader(
    "Upload a chest X-ray (PNG or JPG)", type=["png", "jpg", "jpeg"]
)

if uploaded is None:
    st.info("Upload a CXR image to get started.")
    st.stop()

if not model_loaded:
    st.error("Train a model first and place the checkpoint in `results/checkpoints/`.")
    st.stop()

# ── Preprocess ────────────────────────────────────────────────────────────────

pil_img = Image.open(uploaded).convert("RGB")
transform = val_transforms(224)
tensor    = transform(pil_img)                     # (3, 224, 224)
batch     = tensor.unsqueeze(0)                    # (1, 3, 224, 224)
display   = tensor_to_display(tensor)              # (224, 224, 3) for display

# ── Inference ─────────────────────────────────────────────────────────────────

with torch.no_grad():
    logits = model(batch.to(DEVICE))               # (1, 14)
probs = torch.sigmoid(logits).squeeze().cpu().numpy()   # (14,)

# ── Predictions panel ─────────────────────────────────────────────────────────

st.subheader("Predicted Pathologies")

pos_classes = [(NIH14_CLASSES[i], float(probs[i]))
               for i in range(14) if probs[i] >= conf_threshold]
pos_classes.sort(key=lambda x: x[1], reverse=True)

if pos_classes:
    cols = st.columns(min(len(pos_classes), 4))
    for idx, (cls, prob) in enumerate(pos_classes):
        with cols[idx % 4]:
            st.metric(label=cls, value=f"{prob:.1%}")
else:
    st.success("No pathology exceeds the confidence threshold — likely normal.")

with st.expander("All class probabilities"):
    import pandas as pd
    df_probs = pd.DataFrame({
        "Pathology": NIH14_CLASSES,
        "Probability": [f"{p:.1%}" for p in probs],
        "Raw score": [f"{p:.4f}" for p in probs],
    })
    st.dataframe(df_probs, use_container_width=True)

# ── XAI heatmaps ─────────────────────────────────────────────────────────────

if not selected_methods:
    st.warning("Select at least one XAI method in the sidebar.")
    st.stop()

# Target class = highest predicted class
target_cls = int(np.argmax(probs))
target_name = NIH14_CLASSES[target_cls]
st.subheader(f"XAI Heatmaps — target class: **{target_name}** ({probs[target_cls]:.1%})")

heatmaps: dict[str, np.ndarray] = {}

with st.spinner("Generating heatmaps…"):
    for method_display in selected_methods:
        key = METHOD_KEY[method_display]
        if key in ("gradcam_plus_plus", "hirescam"):
            h = compute_cam_batch(key, model, batch, [target_cls], DEVICE)[0]
        elif key == "integrated_gradients":
            h = compute_integrated_gradients(
                model, batch, [target_cls], DEVICE, n_steps=50
            )[0]
        else:
            h = compute_attention_rollout(model, batch, DEVICE)[0]
        heatmaps[method_display] = h

# Display original + one column per method
n_cols = 1 + len(selected_methods)
cols = st.columns(n_cols)

with cols[0]:
    st.image(display, caption="Original CXR", use_container_width=True)

for idx, (method_display, heatmap) in enumerate(heatmaps.items()):
    overlay = overlay_heatmap(display, heatmap, alpha=0.4)
    with cols[idx + 1]:
        st.image(overlay, caption=method_display, use_container_width=True)

# ── Quantitative metrics (per-image summary) ──────────────────────────────────

if show_metrics and heatmaps:
    st.subheader("Per-Image Quantitative Metrics")
    st.caption(
        "Computed on this single image. Full-dataset results are in "
        "`results/metrics/xai_comparison_table.csv`."
    )

    metric_rows = []
    first_heatmap = next(iter(heatmaps.values()))
    heatmap_np = np.stack(list(heatmaps.values()))                  # (M, H, W)
    n_methods = len(heatmaps)
    class_indices_rep = [target_cls] * n_methods

    with st.spinner("Computing deletion / insertion AUC…"):
        del_auc, ins_auc = compute_deletion_insertion(
            model,
            batch.repeat(n_methods, 1, 1, 1),
            heatmap_np,
            class_indices_rep,
            device=DEVICE,
            n_steps=5,
        )

    for i, (method_display, heatmap) in enumerate(heatmaps.items()):

        def heatmap_fn(imgs, _m=method_display, _k=METHOD_KEY[method_display]):
            if _k in ("gradcam_plus_plus", "hirescam"):
                return compute_cam_batch(_k, model, imgs, [target_cls] * imgs.shape[0], DEVICE)
            elif _k == "integrated_gradients":
                return compute_integrated_gradients(
                    model, imgs, [target_cls] * imgs.shape[0], DEVICE, n_steps=10
                )
            else:
                return compute_attention_rollout(model, imgs, DEVICE)

        rho = compute_spearman_stability(heatmap_fn, batch, noise_std=0.1, n_runs=2)

        metric_rows.append({
            "Method":          method_display,
            "Deletion AUC ↓":  f"{del_auc[i]:.4f}",
            "Insertion AUC ↑": f"{ins_auc[i]:.4f}",
            "Spearman ρ ↑":    f"{rho[0]:.4f}",
        })

    import pandas as pd
    st.dataframe(pd.DataFrame(metric_rows), use_container_width=True)

st.divider()
st.caption(
    "Model trained on NIH ChestX-ray14. "
    "Not validated for clinical use. Research prototype only."
)
