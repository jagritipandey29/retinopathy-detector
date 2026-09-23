"""
Diabetic Retinopathy Screening App

Model: EfficientNet-B4 (5-class DR grading), QWK-selected checkpoint.
Headline: Referable DR (Yes/No) | Charts: stage-probability + confidence donut
Explainability: Grad-CAM "AI attention map"
"""

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import gdown

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_qwk_final.pth")
GDRIVE_FILE_ID = "1f4mtlffwi9omaJKERX-KRY4hrk86B5Uz"
IMG_SIZE = 380
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}
CLASS_COLORS = {
    0: "#2ecc71",
    1: "#a3d977",
    2: "#f5c518",
    3: "#f39c12",
    4: "#e74c3c",
}
REFERABLE_CLASSES = {2, 3, 4}

st.set_page_config(page_title="DR Screening", page_icon="🩺", layout="wide")

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
   .block-container { padding-top: 2rem; }
   .badge-yes {
        background: linear-gradient(135deg, #ff6b6b 0%, #c0392b 100%);
        border-radius: 16px; padding: 28px; text-align: center; color: white;
        box-shadow: 0 8px 24px rgba(192,57,43,0.35);
    }
   .badge-no {
        background: linear-gradient(135deg, #2ecc71 0%, #1a7d3a 100%);
        border-radius: 16px; padding: 28px; text-align: center; color: white;
        box-shadow: 0 8px 24px rgba(26,125,58,0.35);
    }
   .badge-yes h1,.badge-no h1 { margin: 0; font-size: 2rem; }
   .badge-yes p,.badge-no p { margin: 4px 0 0 0; opacity: 0.95; }
   .info-card {
        background: #f8f9fb; border-radius: 12px; padding: 16px 20px;
        border: 1px solid #eaeaea; margin-top: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Model loading - FIXED for Streamlit Cloud
# --------------------------------------------------------------------------
def ensure_model_downloaded():
    # Agar file hai aur size 5MB se bada hai to sahi hai
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 5000000:
        return True
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    if not GDRIVE_FILE_ID:
        return False
    try:
        with st.spinner("Downloading model weights (first run only, ~75MB)..."):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
        return os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 5000000
    except Exception as e:
        st.error(f"Model download failed: {e}. Check Drive link is 'Anyone with the link'")
        return False

@st.cache_resource
def load_model():
    if not ensure_model_downloaded():
        return None
    model = models.efficientnet_b4(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def preprocess(pil_img: Image.Image):
    img = np.array(pil_img.convert("RGB"))
    tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize(), ToTensorV2()])
    tensor = tf(image=img)["image"]
    return tensor.unsqueeze(0), cv2.resize(img, (IMG_SIZE, IMG_SIZE))

@torch.no_grad()
def predict(model, tensor):
    logits = model(tensor.to(DEVICE))
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_class = int(np.argmax(probs))
    referable_prob = float(sum(probs[c] for c in REFERABLE_CLASSES))
    is_referable = pred_class in REFERABLE_CLASSES
    return pred_class, probs, is_referable, referable_prob

# --------------------------------------------------------------------------
# Grad-CAM
# --------------------------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx):
        input_tensor = input_tensor.to(DEVICE)
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[0, class_idx].backward()

        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0].clone()
        for i in range(activations.shape[0]):
            activations[i] *= pooled_grads[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        return heatmap

def overlay_heatmap(base_img_rgb, heatmap, alpha=0.45):
    heatmap_resized = cv2.resize(heatmap, (base_img_rgb.shape[1], base_img_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(base_img_rgb, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

@st.cache_resource
def get_gradcam(_model):
    target_layer = _model.features[-1]
    return GradCAM(_model, target_layer)

# --------------------------------------------------------------------------
# Chart builders
# --------------------------------------------------------------------------
def build_probability_bar(probs):
    labels = [CLASS_NAMES[c] for c in range(5)]
    values = [probs[c] * 100 for c in range(5)]
    colors = [CLASS_COLORS[c] for c in range(5)]
    fig = go.Figure(
        go.Bar(
            x=values, y=labels, orientation="h",
            marker=dict(color=colors),
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Stage-wise confidence (Accuracy Graph)",
        xaxis_title="Probability (%)",
        xaxis=dict(range=[0, 100]),
        height=320,
        margin=dict(l=10, r=30, t=50, b=10),
        plot_bgcolor="white",
    )
    return fig

def build_confidence_donut(referable_prob, is_referable):
    non_ref = 100 - referable_prob * 100
    ref = referable_prob * 100
    colors = ["#e74c3c", "#2ecc71"]
    fig = go.Figure(
        go.Pie(
            labels=["Referable", "Non-referable"],
            values=[ref, non_ref],
            hole=0.65,
            marker=dict(colors=colors),
            textinfo="percent",
            sort=False,
        )
    )
    center_text = "REFERABLE" if is_referable else "SAFE"
    fig.update_layout(
        title="Referable vs Non-referable",
        annotations=[dict(text=center_text, x=0.5, y=0.5, font_size=16, showarrow=False)],
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=True,
    )
    return fig

# --------------------------------------------------------------------------
# UI - FIXED width='stretch'
# --------------------------------------------------------------------------
st.title("🩺 Diabetic Retinopathy Screening")
st.caption(
    "Research/screening-support tool — not a diagnostic device. "
    "Always confirm with an eye-care professional."
)

model = load_model()

if model is None:
    st.error(
        "Could not load model weights. Check Drive file is shared as 'Anyone with the link' "
        "or manually place `best_qwk_final.pth` next to app.py"
    )
    st.stop()

uploaded = st.file_uploader("Upload a retinal fundus image", type=["jpg", "jpeg", "png"])

if uploaded:
    pil_image = Image.open(uploaded)
    tensor, resized_rgb = preprocess(pil_image)

    with st.spinner("Analyzing image..."):
        pred_class, probs, is_referable, referable_prob = predict(model, tensor)
        gradcam = get_gradcam(model)
        heatmap = gradcam.generate(tensor.clone(), pred_class)
        overlay_img = overlay_heatmap(resized_rgb, heatmap)

    if is_referable:
        st.markdown(
            f"""<div class="badge-yes">
                    <h1>⚠️ Referable DR: YES</h1>
                    <p>Refer to an ophthalmologist — confidence {referable_prob*100:.1f}%</p>
                </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""<div class="badge-no">
                    <h1>✅ Referable DR: NO</h1>
                    <p>Routine monitoring — confidence {(1-referable_prob)*100:.1f}%</p>
                </div>""",
            unsafe_allow_html=True,
        )

    st.write("")
    tab_result, tab_attention = st.tabs(["📊 Result & Charts", "🔍 AI Attention Map"])

    with tab_result:
        col_img, col_charts = st.columns([1, 1.3])
        with col_img:
            st.image(pil_image, caption="Uploaded fundus image", width='stretch')
            st.markdown(
                f"""<div class="info-card">
                        <b>Predicted grade:</b> {CLASS_NAMES[pred_class]} (class {pred_class})<br>
                        <b>Model confidence (top class):</b> {probs[pred_class]*100:.1f}%
                    </div>""",
                unsafe_allow_html=True,
            )
        with col_charts:
            st.plotly_chart(build_probability_bar(probs), width='stretch')
            st.plotly_chart(build_confidence_donut(referable_prob, is_referable), width='stretch')

    with tab_attention:
        st.write(
            "Grad-CAM shows which regions influenced the prediction — red/yellow = high influence"
        )
        alpha = st.slider("Heatmap intensity", 0.0, 0.9, 0.45, 0.05)
        overlay_display = overlay_heatmap(resized_rgb, heatmap, alpha=alpha)
        c1, c2 = st.columns(2)
        with c1:
            st.image(resized_rgb, caption="Original (preprocessed)", width='stretch')
        with c2:
            st.image(overlay_display, caption="AI attention map", width='stretch')

    with st.expander("Raw probability table"):
        for c in range(5):
            st.write(f"**{CLASS_NAMES[c]}**: {probs[c]*100:.2f}%")

else:
    st.info("Upload a fundus image to get prediction, charts, and attention map.")
