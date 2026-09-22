import os
import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as tv_models
import streamlit as st
import gdown
import requests
import cv2
from PIL import Image
from efficientnet_pytorch import EfficientNet as LegacyEfficientNet

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
MODEL_PATH = "best_model.pth"
DRIVE_FILE_ID = "1t0FecrXJeVAAqaqpmmpcP72XIhlPpBg4"
NUM_CLASSES = 5
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
MIN_VALID_SIZE_BYTES = 5_000_000

# torchvision EfficientNet variants: constructor fn + native training resolution
TV_VARIANTS = [
    ("efficientnet_b0", tv_models.efficientnet_b0, 224),
    ("efficientnet_b1", tv_models.efficientnet_b1, 240),
    ("efficientnet_b2", tv_models.efficientnet_b2, 260),
    ("efficientnet_b3", tv_models.efficientnet_b3, 300),
    ("efficientnet_b4", tv_models.efficientnet_b4, 380),
    ("efficientnet_b5", tv_models.efficientnet_b5, 456),
    ("efficientnet_b6", tv_models.efficientnet_b6, 528),
    ("efficientnet_b7", tv_models.efficientnet_b7, 600),
]
# efficientnet-pytorch (lukemelas) variants, kept as a fallback family
LEGACY_VARIANTS = [
    ("efficientnet-b0", 224),
    ("efficientnet-b3", 300),
    ("efficientnet-b4", 380),
    ("efficientnet-b5", 456),
]

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
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={"id": file_id}, stream=True, timeout=60)
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
    if token is None:
        for line in response.text.splitlines():
            if "confirm=" in line and "download" in line:
                start = line.find("confirm=") + len("confirm=")
                end = line.find("&", start)
                token = line[start:end if end != -1 else None]
                break
    if token:
        response = session.get(URL, params={"id": file_id, "confirm": token}, stream=True, timeout=60)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)


def _try_gdown(file_id, destination):
    try:
        gdown.download(id=file_id, output=destination, quiet=False, fuzzy=True)
        return
    except TypeError:
        pass
    try:
        gdown.download(id=file_id, output=destination, quiet=False)
        return
    except TypeError:
        pass
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
            f"Downloaded 'best_model.pth' is only {size} bytes and looks like an HTML page. "
            "Google Drive is blocking the automated download — share the file as 'Anyone with "
            "the link', or commit the .pth directly into the GitHub repo instead."
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


def _remap_se_keys(state_dict):
    """This checkpoint's squeeze-excitation blocks are named fc1/fc3
    (fc2 being the parameter-free activation in between), while
    torchvision's SqueezeExcitation module names them fc1/fc2. Remap so
    the excite conv lines up with torchvision's second conv layer."""
    return {k.replace(".fc3.", ".fc2."): v for k, v in state_dict.items()}


# ----------------------------------------------------------------------
# MODEL LOADING — tries torchvision EfficientNet variants first (this
# checkpoint's "features.N.block..." keys match that family), then
# falls back to the efficientnet-pytorch family. Fails loudly rather
# than silently keeping random weights.
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    download_model()
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    diagnostics = {"checkpoint_type": str(type(checkpoint))}
    if isinstance(checkpoint, dict):
        diagnostics["checkpoint_keys_sample"] = list(checkpoint.keys())[:10]

    state_dict, ready_model = _extract_state_dict(checkpoint)

    if ready_model is not None:
        ready_model.eval()
        st.session_state["_detected_arch"] = "loaded as full nn.Module"
        st.session_state["_model_family"] = "module"
        st.session_state["_model_resolution"] = 380
        st.session_state["_load_diagnostics"] = diagnostics
        return ready_model

    if state_dict is None or len(state_dict) == 0:
        raise RuntimeError(f"Checkpoint parsed but contained no weights. Diagnostics: {diagnostics}")

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    diagnostics["num_state_dict_keys"] = len(state_dict)

    results = {}
    best = {"missing": None, "model": None, "family": None, "arch": None, "resolution": None}

    # --- Family 1: torchvision EfficientNet (features.N.block...) ---
    remapped = _remap_se_keys(state_dict)
    for arch_name, ctor, resolution in TV_VARIANTS:
        try:
            candidate = ctor(weights=None, num_classes=NUM_CLASSES)
            missing, unexpected = candidate.load_state_dict(remapped, strict=False)
            critical_missing = [k for k in missing if "classifier" not in k]
            results[f"torchvision:{arch_name}"] = {
                "missing": len(missing),
                "critical_missing": len(critical_missing),
                "unexpected": len(unexpected),
            }
            if best["missing"] is None or len(critical_missing) < best["missing"]:
                best.update(
                    missing=len(critical_missing),
                    model=candidate,
                    family="torchvision",
                    arch=arch_name,
                    resolution=resolution,
                )
        except Exception as e:
            results[f"torchvision:{arch_name}"] = {"error": str(e)}

    # --- Family 2: efficientnet-pytorch (legacy) fallback ---
    for arch_name, resolution in LEGACY_VARIANTS:
        try:
            candidate = LegacyEfficientNet.from_name(arch_name, num_classes=NUM_CLASSES)
            missing, unexpected = candidate.load_state_dict(state_dict, strict=False)
            critical_missing = [k for k in missing if not k.startswith("_fc")]
            results[f"legacy:{arch_name}"] = {
                "missing": len(missing),
                "critical_missing": len(critical_missing),
                "unexpected": len(unexpected),
            }
            if best["missing"] is None or len(critical_missing) < best["missing"]:
                best.update(
                    missing=len(critical_missing),
                    model=candidate,
                    family="legacy",
                    arch=arch_name,
                    resolution=resolution,
                )
        except Exception as e:
            results[f"legacy:{arch_name}"] = {"error": str(e)}

    diagnostics["per_arch_results"] = results
    st.session_state["_load_diagnostics"] = diagnostics

    total_keys = len(state_dict)
    if best["model"] is None or best["missing"] is None or best["missing"] > max(5, 0.02 * total_keys):
        raise RuntimeError(
            "Could not confidently match the checkpoint to any known EfficientNet variant "
            f"(torchvision or efficientnet-pytorch) — best candidate "
            f"'{best['family']}:{best['arch']}' still had {best['missing']} unmatched keys. "
            f"See diagnostics below. Details: {results}"
        )

    best["model"].eval()
    st.session_state["_detected_arch"] = f"{best['family']}:{best['arch']} (missing critical keys: {best['missing']})"
    st.session_state["_model_family"] = best["family"]
    st.session_state["_model_resolution"] = best["resolution"]
    return best["model"]


# ----------------------------------------------------------------------
# PREPROCESSING — resolution depends on which variant matched, but the
# resize+centercrop ratio still protects against WhatsApp/screenshot
# black-border images the way the fixed 256/224 pipeline did.
# ----------------------------------------------------------------------
def preprocess_image(pil_image, resolution):
    pil_image = pil_image.convert("RGB")
    resize_dim = int(round(resolution * 256 / 224))
    transform = transforms.Compose(
        [
            transforms.Resize((resize_dim, resize_dim)),
            transforms.CenterCrop(resolution),
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


def get_target_layer(model, family):
    if family == "torchvision":
        return model.features[-1]
    if family == "legacy":
        if hasattr(model, "_conv_head"):
            return model._conv_head
        return model._blocks[-1]
    # unknown / plain nn.Module fallback: use the last child module with parameters
    last = None
    for m in model.modules():
        if isinstance(m, (nn.Conv2d,)):
            last = m
    return last


def overlay_gradcam(pil_image, cam, resolution):
    img = np.array(pil_image.resize((resolution, resolution))).astype(np.float32) / 255.0
    heatmap = cv2.resize(cam, (resolution, resolution))
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

    lines = [
        f"**Prediction summary:** The model identified this fundus image as **{CLASS_NAMES[pred]}** "
        f"with **{conf:.1f}%** confidence.",
        f"**Typical findings at this stage:** {CLASS_NAMES[pred]} is usually characterized by "
        f"{CLASS_DESCRIPTIONS[pred]}.",
    ]

    if hot_frac > 0.5:
        lines.append(
            f"**Where the model looked:** Grad-CAM shows attention concentrated mainly in the "
            f"**{region}** of the retina (~{hot_frac:.0f}% of the image showed strong activation). "
            "Compare that region of the original image for hemorrhages, exudates, or vessel abnormalities."
        )
    else:
        lines.append(
            "**Where the model looked:** Attention was fairly diffuse rather than sharply "
            "localized, typical for 'No DR' or very early/subtle findings."
        )

    if runner_gap < 15:
        lines.append(
            f"**Borderline case:** The second-most-likely class, **{CLASS_NAMES[runner_up]}** "
            f"({probs_np[runner_up]*100:.1f}%), is close to the top prediction (gap: {runner_gap:.1f} "
            "points). Treat this as borderline and prioritize clinical correlation."
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

model_family = st.session_state.get("_model_family", "torchvision")
resolution = st.session_state.get("_model_resolution", 380)
detected_arch = st.session_state.get("_detected_arch", "unknown")
st.caption(f"Backbone detected: `{detected_arch}` · input resolution: {resolution}px")
with st.expander("Model load diagnostics"):
    st.json(st.session_state.get("_load_diagnostics", {}))

uploaded_file = st.file_uploader(
    "Upload a fundus image", type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    raw_image = Image.open(io.BytesIO(uploaded_file.read()))
    display_image, input_tensor = preprocess_image(raw_image, resolution)

    if st.button("Predict", type="primary"):
        with st.spinner("Running inference..."):
            with torch.no_grad():
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1)[0]
                pred = int(torch.argmax(probs).item())
                conf = float(probs[pred].item()) * 100

            target_layer = get_target_layer(model, model_family)
            gradcam_ok = False
            resized_heatmap = None
            if target_layer is not None:
                cam_extractor = GradCAM(model, target_layer)
                try:
                    cam = cam_extractor.generate(input_tensor, pred)
                    overlay, resized_heatmap = overlay_gradcam(display_image, cam, resolution)
                    gradcam_ok = True
                except Exception as e:
                    gradcam_error = str(e)
                finally:
                    cam_extractor.remove_hooks()
            else:
                gradcam_error = "No suitable target layer found for Grad-CAM on this model."

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
