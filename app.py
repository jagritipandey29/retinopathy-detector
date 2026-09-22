import os
import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import streamlit as st
import gdown
import requests
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
CANDIDATE_ARCHS = ["efficientnet-b4", "efficientnet-b3", "efficientnet-b0", "efficientnet-b5"]
MIN_VALID_SIZE_BYTES = 5_000_000  # a real EfficientNet checkpoint is tens of MB

CLASS_DESCRIPTIONS = {
    0: "no visible microaneurysms, hemorrhages, or exudates",
    1: "a small number of microaneurysms (tiny red dots) — typically the earliest visible sign of retinal damage",
    2: "more numerous microaneurysms together with early hemorrhages and/or hard exudates",
    3: "extensive hemorrhages across multiple retinal quadrants, venous beading, and/or intraretinal microvascular abnormalities",
    4: "neovascularization (abnormal new blood vessel growth) and/or vitreous/preretinal hemorrhage, indicating advanced disease",
}

st.set_page_config(page_title="Diabetic Retinopathy Detector", layout="wide")


# ----------------------------------------------------------------------
# MODEL DOWNLOAD — version-proof against gdown API changes
# ----------------------------------------------------------------------
def _looks_like_html(path):
    try:
        with open(path, "rb") as f:
            head = f.read(200).lstrip()
        return head.startswith(b"<") or b"<html" in head.lower()
    except Exception:
        return False


def _manual_drive_download(file_id, destination):
    """Fallback that bypasses gdown entirely using the classic confirm-token
    dance for large Google Drive files. Works regardless of gdown version."""
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={"id": file_id}, stream=True, timeout=60)

    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
    if token is None:
        # Newer Drive UI sometimes embeds the confirm token in the HTML body
        for line in response.text.splitlines():
            if "confirm=" in line and "download" in line:
                start = line.find("confirm=") + len("confirm=")
                end = line.find("&", start)
                token = line[start:end if end != -1 else None]
                break

    if token:
        response = session.get(
            URL, params={"id": file_id, "confirm": token}, stream=True, timeout=60
        )

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)


def _try_gdown(file_id, destination):
    """Call gdown in whatever way its installed version supports."""
    try:
        gdown.download(id=file_id, output=destination, quiet=False, fuzzy=True)
        return
    except TypeError:
        pass  # installed gdown predates the `fuzzy` kwarg
    try:
        gdown.download(id=file_id, output=destination, quiet=False)
        return
    except TypeError:
        pass  # very old gdown: no `id=` kwarg either
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=False)


def download_model():
    need_download = not os.path.exists(MODEL_PATH)
    if not need_download:
        size = os.path.getsize(MODEL_PATH)
        if size < MIN_VALID_SIZE_BYTES or _looks_like_html(MODEL_PATH):
            os.remove(MODEL_PATH)
            need_download = True

    if need_download:
        with st.spinner("Downloading model weights (first run only)..."):
            try:
                _try_gdown(DRIVE_FILE_ID, MODEL_PATH)
            except Exception:
                pass

            valid = (
                os.path.exists(MODEL_PATH)
                and os.path.getsize(MODEL_PATH) >= MIN_VALID_SIZE_BYTES
                and not _looks_like_html(MODEL_PATH)
            )
            if not valid:
                if os.path.exists(MODEL_PATH):
                    os.remove(MODEL_PATH)
                _manual_drive_download(DRIVE_FILE_ID, MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model download failed — no file was written.")

    size = os.path.getsize(MODEL_PATH)
    if size < MIN_VALID_SIZE_BYTES or _looks_like_html(MODEL_PATH):
        os.remove(MODEL_PATH)
        raise RuntimeError(
            f"Downloaded 'best_model.pth' is only {size} bytes and looks like an HTML page, "
            "not model weights. Google Drive is blocking the automated download for this file. "
            "Fix: make sure the Drive file is shared as 'Anyone with the link', or download the "
            ".pth manually and commit it directly into the GitHub repo (Streamlit Cloud will "
            "then skip the runtime download entirely)."
        )
    return size


# ----------------------------------------------------------------------
# CHECKPOINT PARSING
# ----------------------------------------------------------------------
def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, nn.Module):
        return None, checkpoint

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model_state", "model"):
            if key in checkpoint:
                inner = checkpoint[key]
                if isinstance(inner, nn.Module):
                    return None, inner
                if isinstance(inner, dict):
                    return inner, None
        if len(checkpoint) > 0 and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint, None

    raise ValueError(f"Unrecognized checkpoint format: {type(checkpoint)}")


# ----------------------------------------------------------------------
# MODEL LOADING — fails loudly instead of silently falling back to
# random weights (the cause of the earlier uniform-20% bug).
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    download_model()
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    diagnostics = {"checkpoint_type": str(type(checkpoint))}
    if isinstance(checkpoint, dict):
        diagnostics["checkpoint_keys"] = list(checkpoint.keys())[:20]

    state_dict, ready_model = _extract_state_dict(checkpoint)

    if ready_model is not None:
        ready_model.eval()
        st.session_state["_detected_arch"] = "loaded as full nn.Module (no arch guessing needed)"
        st.session_state["_load_diagnostics"] = diagnostics
        return ready_model

    if state_dict is None or len(state_dict) == 0:
        raise RuntimeError(f"Checkpoint parsed but contained no weights. Diagnostics: {diagnostics}")

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    diagnostics["num_state_dict_keys"] = len(state_dict)

    results_per_arch = {}
    best_arch, best_model, best_missing = None, None, None

    for arch in CANDIDATE_ARCHS:
        try:
            candidate = EfficientNet.from_name(arch, num_classes=NUM_CLASSES)
            missing, unexpected = candidate.load_state_dict(state_dict, strict=False)
            critical_missing = [k for k in missing if not k.startswith("_fc")]
            results_per_arch[arch] = {
                "missing": len(missing),
                "critical_missing": len(critical_missing),
                "unexpected": len(unexpected),
            }
            if best_missing is None or len(critical_missing) < best_missing:
                best_arch, best_model, best_missing = arch, candidate, len(critical_missing)
        except Exception as e:
            results_per_arch[arch] = {"error": str(e)}
            continue

    diagnostics["per_arch_results"] = results_per_arch
    st.session_state["_load_diagnostics"] = diagnostics

    total_backbone_keys = len(state_dict)
    if best_model is None or best_missing is None or best_missing > max(5, 0.02 * total_backbone_keys):
        raise RuntimeError(
            "Could not confidently match the checkpoint to any known EfficientNet "
            f"architecture — best candidate ('{best_arch}') still had {best_missing} "
            f"unmatched backbone keys. Details: {results_per_arch}"
        )

    best_model.eval()
    st.session_state["_detected_arch"] = f"{best_arch} (missing critical keys: {best_missing})"
    return best_model


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

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2))

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
    if hasattr(model, "_conv_head"):
        return model._conv_head
    return model._blocks[-1]


def overlay_gradcam(pil_image_224, cam):
    img = np.array(pil_image_224.resize((224, 224))).astype(np.float32) / 255.0
    heatmap = cv2.resize(cam, (224, 224))
    heatmap_u8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = 0.5 * heatmap_color + 0.5 * img
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8), heatmap


# ----------------------------------------------------------------------
# AI-STYLE ANALYSIS REPORT
# ----------------------------------------------------------------------
def describe_attention_region(heatmap):
    """Summarize WHERE the model focused, from the Grad-CAM heatmap."""
    h, w = heatmap.shape
    yy, xx = np.mgrid[0:h, 0:w]
    total = heatmap.sum()
    if total <= 1e-6:
        return "no single concentrated region (attention was diffuse across the image)", 0.0

    cy = (yy * heatmap).sum() / total
    cx = (xx * heatmap).sum() / total

    vert = "upper" if cy < h / 3 else ("lower" if cy > 2 * h / 3 else "central")
    horiz = "left" if cx < w / 3 else ("right" if cx > 2 * w / 3 else "central")

    if vert == "central" and horiz == "central":
        region = "central (macular) region"
    elif vert == "central":
        region = f"{horiz} side, centered vertically"
    elif horiz == "central":
        region = f"{vert} field, centered horizontally"
    else:
        region = f"{vert}-{horiz} field"

    hot_frac = float((heatmap > 0.6).mean()) * 100
    return region, hot_frac


def generate_ai_explanation(pred, probs_np, heatmap):
    region, hot_frac = describe_attention_region(heatmap)
    conf = probs_np[pred] * 100

    ranked = sorted(range(len(CLASS_NAMES)), key=lambda i: probs_np[i], reverse=True)
    runner_up = ranked[1]
    runner_gap = (probs_np[ranked[0]] - probs_np[runner_up]) * 100

    lines = []
    lines.append(
        f"**Prediction summary:** The model identified this fundus image as **{CLASS_NAMES[pred]}** "
        f"with **{conf:.1f}%** confidence."
    )
    lines.append(
        f"**Typical findings at this stage:** {CLASS_NAMES[pred]} is usually characterized by "
        f"{CLASS_DESCRIPTIONS[pred]}."
    )

    if hot_frac > 0.5:
        lines.append(
            f"**Where the model looked:** Grad-CAM shows the model's attention was concentrated "
            f"mainly in the **{region}** of the retina (roughly {hot_frac:.0f}% of the image area "
            f"showed strong activation). This is the region driving the prediction — worth comparing "
            f"against that area of the original image for hemorrhages, exudates, or vessel abnormalities."
        )
    else:
        lines.append(
            "**Where the model looked:** Attention was fairly diffuse across the retina rather than "
            "sharply localized, which is typical for 'No DR' or very early/subtle findings."
        )

    if runner_gap < 15:
        lines.append(
            f"**Borderline case:** The second-most-likely class, **{CLASS_NAMES[runner_up]}** "
            f"({probs_np[runner_up]*100:.1f}%), is close to the top prediction (gap: {runner_gap:.1f} "
            "points). Treat this case as borderline and prioritize clinical correlation rather than "
            "relying on the label alone."
        )
    else:
        lines.append(
            f"**Confidence spread:** The gap to the next most likely class "
            f"({CLASS_NAMES[runner_up]}, {probs_np[runner_up]*100:.1f}%) is {runner_gap:.1f} points, "
            "indicating a fairly decisive prediction."
        )

    lines.append(
        "_This explanation is generated automatically from the model's output probabilities and its "
        "Grad-CAM attention map — it is a screening aid, not a clinical diagnosis. Always confirm with "
        "a qualified ophthalmologist._"
    )
    return "\n\n".join(lines)


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("🩺 Diabetic Retinopathy Detection")
st.caption("EfficientNet-based classifier trained on the APTOS 2019 dataset")

try:
    with st.spinner("Loading model..."):
        model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    diag = st.session_state.get("_load_diagnostics")
    if diag:
        with st.expander("Debug details"):
            st.json(diag)
    st.stop()

detected_arch = st.session_state.get("_detected_arch", "unknown")
st.caption(f"Backbone detected: `{detected_arch}`")
with st.expander("Model load diagnostics"):
    st.json(st.session_state.get("_load_diagnostics", {}))

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

            target_layer = get_target_layer(model)
            cam_extractor = GradCAM(model, target_layer)
            try:
                cam = cam_extractor.generate(input_tensor, pred)
                overlay, resized_heatmap = overlay_gradcam(display_image, cam)
                gradcam_ok = True
            except Exception as e:
                gradcam_ok = False
                gradcam_error = str(e)
                resized_heatmap = None
            finally:
                cam_extractor.remove_hooks()

        probs_np = probs.detach().numpy()
        if probs_np.std() < 0.02:
            st.warning(
                "⚠️ All class probabilities are nearly identical — this usually means the loaded "
                "weights are still (partially) untrained. Check 'Model load diagnostics' above."
            )

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

        st.write("### 🧠 AI Analysis")
        if gradcam_ok and resized_heatmap is not None:
            st.markdown(generate_ai_explanation(pred, probs_np, resized_heatmap))
        else:
            st.info("AI analysis requires Grad-CAM output, which failed for this image (see warning above).")

        st.caption(
            "This tool is a screening aid, not a diagnostic device. "
            "Always confirm results with a qualified ophthalmologist."
        )
else:
    st.info("Upload a fundus image to get a prediction.")
