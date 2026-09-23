"""
Diabetic Retinopathy Screening App
-----------------------------------
Model: EfficientNet-B4 (5-class DR grading), warm-started + fine-tuned with QWK selection.
Headline: Referable DR (Yes/No)  |  Detail: full 5-stage breakdown.

Run:
    pip install streamlit torch torchvision albumentations opencv-python-headless pillow numpy
    streamlit run app.py

Place `best_qwk_final.pth` in the same folder as this file (or edit MODEL_PATH below).
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
import gdown

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_qwk_final.pth")
# Google Drive file ID of best_qwk_final.pth (uploaded, link-shared "Anyone with the link")
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
# 0,1 = non-referable | 2,3,4 = referable (needs ophthalmologist referral)
REFERABLE_CLASSES = {2, 3, 4}

st.set_page_config(page_title="DR Screening", page_icon="🩺", layout="centered")

# --------------------------------------------------------------------------
# Model loading (cached so it only loads once per session)
# --------------------------------------------------------------------------
def ensure_model_downloaded():
    """Download best_qwk_final.pth from Google Drive on first run if it's not already on disk."""
    if os.path.exists(MODEL_PATH):
        return True
    if not GDRIVE_FILE_ID:
        return False
    try:
        with st.spinner("Downloading model weights (first run only, ~70-80MB)..."):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
        return os.path.exists(MODEL_PATH)
    except Exception as e:
        st.error(f"Model download failed: {e}")
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


def preprocess(pil_img: Image.Image) -> torch.Tensor:
    img = np.array(pil_img.convert("RGB"))
    tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize(), ToTensorV2()])
    tensor = tf(image=img)["image"]
    return tensor.unsqueeze(0)


@torch.no_grad()
def predict(model, pil_img: Image.Image):
    x = preprocess(pil_img).to(DEVICE)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_class = int(np.argmax(probs))
    referable_prob = float(sum(probs[c] for c in REFERABLE_CLASSES))
    is_referable = pred_class in REFERABLE_CLASSES
    return pred_class, probs, is_referable, referable_prob


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("🩺 Diabetic Retinopathy Screening")
st.caption(
    "Research/screening-support tool — not a diagnostic device. "
    "Always confirm with an eye-care professional."
)

model = load_model()

if model is None:
    st.error(
        "Could not load model weights (auto-download from Google Drive failed). "
        "Check that the Drive file is shared as 'Anyone with the link', "
        "or manually place `best_qwk_final.pth` next to app.py and rerun."
    )
    st.stop()

uploaded = st.file_uploader("Upload a retinal fundus image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Analyzing..."):
        pred_class, probs, is_referable, referable_prob = predict(model, image)

    with col2:
        # ---- Headline: Referable DR badge ----
        if is_referable:
            st.markdown(
                f"""
                <div style="background-color:#ffe3e3;border:2px solid #ff4d4d;
                border-radius:12px;padding:20px;text-align:center;">
                    <h2 style="color:#c00000;margin:0;">⚠️ Referable DR: YES</h2>
                    <p style="margin:6px 0 0 0;">Refer to an ophthalmologist</p>
                    <p style="font-size:14px;color:#555;margin-top:8px;">
                        Confidence: {referable_prob*100:.1f}%
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="background-color:#e3ffe8;border:2px solid #2ecc71;
                border-radius:12px;padding:20px;text-align:center;">
                    <h2 style="color:#1a7d3a;margin:0;">✅ Referable DR: NO</h2>
                    <p style="margin:6px 0 0 0;">Routine monitoring recommended</p>
                    <p style="font-size:14px;color:#555;margin-top:8px;">
                        Confidence: {(1-referable_prob)*100:.1f}%
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.subheader("Detailed 5-stage breakdown")
    st.markdown(f"**Predicted grade:** {CLASS_NAMES[pred_class]} (class {pred_class})")

    # Sorted bar breakdown
    for c in range(5):
        label = CLASS_NAMES[c]
        pct = probs[c] * 100
        st.write(f"{label}")
        st.progress(min(int(pct), 100) / 100)
        st.caption(f"{pct:.1f}%")

    st.markdown("---")
    st.caption(
        "This tool provides a screening-support estimate only. "
        "It is not a substitute for professional medical diagnosis."
    )
else:
    st.info("Upload a fundus image to get a prediction.")
