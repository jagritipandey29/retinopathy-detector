import os
import io
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import streamlit as st
import gdown
import cv2
from PIL import Image
from efficientnet_pytorch import EfficientNet

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
MODEL_PATH = "best_model.pth"
DRIVE_FILE_ID = "1t0FecrXJeVAAqaqpmmpcP72XIhlPpBg4"
NUM_CLASSES = 5
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
CANDIDATE_ARCHS = ["efficientnet-b4", "efficientnet-b3", "efficientnet-b0"]

st.set_page_config(page_title="Diabetic Retinopathy Detector", layout="wide")


# ----------------------------------------------------------------------
# MODEL DOWNLOAD
# ----------------------------------------------------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights (first run only)..."):
            gdown.download(id=DRIVE_FILE_ID, output=MODEL_PATH, quiet=False)


# ----------------------------------------------------------------------
# ARCHITECTURE AUTO-DETECTION
# ----------------------------------------------------------------------
def _extract_state_dict(checkpoint):
    """Handle checkpoints saved as raw state_dict OR wrapped in a dict."""
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        # Might already be a flat state_dict (dict of tensors)
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise ValueError("Unrecognized checkpoint format")


@st.cache_resource(show_spinner=False)
def load_model():
    download_model()
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)

    # Strip common prefixes (e.g. "module." from DataParallel)
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "")
        cleaned[new_k] = v
    state_dict = cleaned

    last_error = None
    for arch in CANDIDATE_ARCHS:
        try:
            model = EfficientNet.from_name(arch, num_classes=NUM_CLASSES)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            # A correct architecture match should have (near) zero missing/
            # unexpected keys for the conv backbone. If too many core keys
            # are missing, this arch is wrong -> try the next one.
            critical_missing = [
                k for k in missing if not k.startswith("_fc")
            ]
            if len(critical_missing) == 0:
                model.eval()
                st.session_state["_detected_arch"] = arch
                return model
        except Exception as e:
            last_error = e
            continue

    # If nothing matched cleanly, fall back to strict=False on the first
    # architecture so the app still runs, but warn the user.
    try:
        model = EfficientNet.from_name(CANDIDATE_ARCHS[0], num_classes=NUM_CLASSES)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        st.session_state["_detected_arch"] = CANDIDATE_ARCHS[0] + " (forced, mismatch)"
        return model
    except Exception as e:
        raise RuntimeError(
            f"Could not load checkpoint into any known architecture. Last error: {last_error or e}"
        )


# ----------------------------------------------------------------------
# PREPROCESSING
# ----------------------------------------------------------------------
def preprocess_image(pil_image):
    pil_image = pil_image.convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    tensor = transform(pil_image).unsqueeze(0)
    return pil_image, tensor


# ----------------------------------------------------------------------
# GRAD-CAM
# ----------------------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self.bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[:, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients[0]      # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        weights = gradients.mean(dim=(1, 2))  # (C,)

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy()

    def remove_hooks(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()


def get_target_layer(model):
    # _conv_head is the last conv layer before pooling -> good spatial
    # resolution with strong semantic features for EfficientNet.
    if hasattr(model, "_conv_head"):
        return model._conv_head
    return model._blocks[-1]


def overlay_gradcam(pil_image_224, cam):
    img = np.array(pil_image_224.resize((224, 224))).astype(np.float32) / 255.0
    heatmap = cv2.resize(cam, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = 0.5 * heatmap_color + 0.5 * img
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("🩺 Diabetic Retinopathy Detection")
st.caption("EfficientNet-based classifier trained on the APTOS 2019 dataset")

with st.spinner("Loading model..."):
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

detected_arch = st.session_state.get("_detected_arch", "unknown")
st.caption(f"Backbone detected: `{detected_arch}`")

uploaded_file = st.file_uploader(
    "Upload a fundus image", type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    raw_image = Image.open(io.BytesIO(uploaded_file.read()))
    display_image, input_tensor = preprocess_image(raw_image)

    if st.button("Predict", type="primary"):
        with st.spinner("Running inference..."):
            with torch.no_grad():
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1)[0]
                pred = int(torch.argmax(probs).item())
                conf = float(probs[pred].item()) * 100

            # ---- Grad-CAM (needs gradients, so run outside no_grad) ----
            target_layer = get_target_layer(model)
            cam_extractor = GradCAM(model, target_layer)
            try:
                cam = cam_extractor.generate(input_tensor, pred)
                overlay = overlay_gradcam(display_image, cam)
                gradcam_ok = True
            except Exception as e:
                gradcam_ok = False
                gradcam_error = str(e)
            finally:
                cam_extractor.remove_hooks()

        col1, col2 = st.columns(2)
        with col1:
            st.image(display_image, caption="Uploaded Image", use_container_width=True)
        with col2:
            if gradcam_ok:
                st.image(overlay, caption="Grad-CAM Explanation", use_container_width=True)
            else:
                st.warning(f"Grad-CAM could not be generated: {gradcam_error}")

        st.subheader(f"Prediction: {CLASS_NAMES[pred]}")
        st.metric("Confidence", f"{conf:.2f}%")

        st.write("### Class probabilities")
        for i, name in enumerate(CLASS_NAMES):
            p = float(probs[i].item()) * 100
            st.write(f"{name}: {p:.2f}%")
            st.progress(min(max(p / 100, 0.0), 1.0))

        st.write("### Recommendation")
        if pred == 0:
            st.success("Result: Healthy — No signs of Diabetic Retinopathy detected.")
        elif pred == 1:
            st.warning("Result: Mild DR — Early stage. Please monitor and get periodic checkups.")
        elif pred == 2:
            st.warning("Result: Moderate DR — Please consult an ophthalmologist.")
        else:
            st.error("Result: Severe / Proliferative DR — Consult a doctor immediately.")

        st.caption(
            "This tool is a screening aid, not a diagnostic device. "
            "Always confirm results with a qualified ophthalmologist."
        )
else:
    st.info("Upload a fundus image to get a prediction.")
