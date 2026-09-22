import os
import io
import numpy as np
import pandas as pd
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
CLASS_ICONS = ["✅", "🟡", "🟠", "🔴", "🟣"]
CLASS_COLORS = ["#22c55e", "#eab308", "#f97316", "#ef4444", "#a855f7"]
MIN_VALID_SIZE_BYTES = 5_000_000

# --- Class ORDER is a guess unless verified against the training notebook ---
# If the model was trained on the original train.csv 'diagnosis' column,
# index order is 0=No DR..4=Proliferative (DIAGNOSIS_ORDER below).
# If it was trained via torchvision.datasets.ImageFolder on folders named
# "Mild"/"Moderate"/"No_DR"/"Proliferate_DR"/"Severe", PyTorch sorts those
# folder names ALPHABETICALLY at train time, giving a totally different
# index->label mapping (ALPHABETICAL_ORDER below). Picking the wrong one
# here will make an otherwise-correct model look like it's misclassifying
# every image. Verify with `print(train_dataset.classes)` in the training
# notebook, then set ACTIVE_ORDER_KEY accordingly (or switch it live in
# the "Class order" debug panel in the app).
CLASS_ORDER_PRESETS = {
    "diagnosis_csv (0=NoDR,1=Mild,2=Moderate,3=Severe,4=Proliferative)":
        ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"],
    "alphabetical_folder (ImageFolder default sort)":
        ["Mild", "Moderate", "No DR", "Proliferative DR", "Severe"],
}
ACTIVE_ORDER_KEY = "diagnosis_csv (0=NoDR,1=Mild,2=Moderate,3=Severe,4=Proliferative)"

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
LEGACY_VARIANTS = [
    ("efficientnet-b0", 224),
    ("efficientnet-b3", 300),
    ("efficientnet-b4", 380),
    ("efficientnet-b5", 456),
]

# Keyed by class NAME (not index) so descriptions stay correct no matter
# which index<->label ordering preset is active.
CLASS_DESCRIPTIONS = {
    "No DR": "no visible microaneurysms, hemorrhages, or exudates",
    "Mild": "a small number of microaneurysms (tiny red dots) — typically the earliest visible sign of retinal damage",
    "Moderate": "more numerous microaneurysms together with early hemorrhages and/or hard exudates",
    "Severe": "extensive hemorrhages across multiple retinal quadrants, venous beading, and/or intraretinal microvascular abnormalities",
    "Proliferative DR": "neovascularization (abnormal new blood vessel growth) and/or vitreous/preretinal hemorrhage, indicating advanced disease",
}

st.set_page_config(page_title="NetraSeva - DR Detection", layout="wide")

st.markdown(
    """
    <style>
    .result-card {
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.03);
    }
    .badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.1rem;
        color: white;
    }
    .conf-sub { opacity: 0.75; font-size: 0.9rem; margin-top: 0.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


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
    return {k.replace(".fc3.", ".fc2."): v for k, v in state_dict.items()}


# ----------------------------------------------------------------------
# MODEL LOADING
#
# IMPORTANT: this function is wrapped in @st.cache_resource, which is a
# PROCESS-WIDE cache shared across every user session — not per-session
# like st.session_state. Its body only runs on the very first call after
# a deploy/restart; every later call (including from brand-new sessions)
# just returns the cached value WITHOUT re-executing this function.
#
# That means anything the model's *identity* depends on (which family it
# is, which target layer Grad-CAM needs, what resolution to preprocess
# at) MUST be returned as part of the cached value itself. Writing it to
# st.session_state instead is a bug: a new session would read its own
# empty/default session_state while silently getting the OTHER family's
# cached model object, causing exactly the AttributeError seen before.
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
        return {
            "model": ready_model,
            "family": "module",
            "arch": "full nn.Module (no arch guessing needed)",
            "resolution": 380,
            "diagnostics": diagnostics,
        }

    if state_dict is None or len(state_dict) == 0:
        raise RuntimeError(f"Checkpoint parsed but contained no weights. Diagnostics: {diagnostics}")

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    diagnostics["num_state_dict_keys"] = len(state_dict)

    results = {}
    best = {"missing": None, "model": None, "family": None, "arch": None, "resolution": None}

    # --- Family 1: efficientnet-pytorch (legacy "_blocks"/"_fc") ---
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
                best.update(missing=len(critical_missing), model=candidate,
                            family="legacy", arch=arch_name, resolution=resolution)
        except Exception as e:
            results[f"legacy:{arch_name}"] = {"error": str(e)}

    # --- Family 2: torchvision EfficientNet ("features.N.block...") ---
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
                best.update(missing=len(critical_missing), model=candidate,
                            family="torchvision", arch=arch_name, resolution=resolution)
        except Exception as e:
            results[f"torchvision:{arch_name}"] = {"error": str(e)}

    diagnostics["per_arch_results"] = results

    total_keys = len(state_dict)
    if best["model"] is None or best["missing"] is None or best["missing"] > max(5, 0.02 * total_keys):
        raise RuntimeError(
            "Could not confidently match the checkpoint to any known EfficientNet variant "
            f"— best candidate '{best['family']}:{best['arch']}' still had {best['missing']} "
            f"unmatched keys. Details: {results}"
        )

    best["model"].eval()
    return {
        "model": best["model"],
        "family": best["family"],
        "arch": f"{best['family']}:{best['arch']} (missing critical keys: {best['missing']})",
        "resolution": best["resolution"],
        "diagnostics": diagnostics,
    }


# ----------------------------------------------------------------------
# PREPROCESSING
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
# TEST-TIME AUGMENTATION
# ----------------------------------------------------------------------
def predict_with_tta(model, base_tensor):
    views = [base_tensor, torch.flip(base_tensor, dims=[3])]  # original + horizontal flip
    with torch.no_grad():
        probs_sum = None
        for v in views:
            logits = model(v)
            p = F.softmax(logits, dim=1)
            probs_sum = p if probs_sum is None else probs_sum + p
        probs = (probs_sum / len(views))[0]
    return probs


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
    """Robust, self-contained detection — does NOT rely on any external
    'which family is this' bookkeeping, so it can't go stale across
    sessions or cache hits. Just inspects the actual model object."""
    if hasattr(model, "_conv_head"):          # efficientnet-pytorch
        return model._conv_head
    if hasattr(model, "_blocks"):             # efficientnet-pytorch (older)
        return model._blocks[-1]
    if hasattr(model, "features"):            # torchvision EfficientNet / others
        return model.features[-1]
    if hasattr(model, "layer4"):              # resnet-style
        return model.layer4[-1]
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv


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


def generate_ai_explanation(pred, probs_np, heatmap, class_names):
    region, hot_frac = describe_attention_region(heatmap)
    conf = probs_np[pred] * 100
    ranked = sorted(range(len(class_names)), key=lambda i: probs_np[i], reverse=True)
    runner_up = ranked[1]
    runner_gap = (probs_np[ranked[0]] - probs_np[runner_up]) * 100

    lines = [
        f"**Prediction summary:** The model identified this fundus image as **{class_names[pred]}** "
        f"with **{conf:.1f}%** confidence.",
        f"**Typical findings at this stage:** {class_names[pred]} is usually characterized by "
        f"{CLASS_DESCRIPTIONS[class_names[pred]]}.",
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
            f"**Borderline case:** The second-most-likely class, **{class_names[runner_up]}** "
            f"({probs_np[runner_up]*100:.1f}%), is close to the top prediction (gap: {runner_gap:.1f} "
            "points). Treat this as borderline and prioritize clinical correlation."
        )
    else:
        lines.append(
            f"**Confidence spread:** The gap to the next most likely class "
            f"({class_names[runner_up]}, {probs_np[runner_up]*100:.1f}%) is {runner_gap:.1f} points, "
            "indicating a fairly decisive prediction."
        )
    lines.append(
        "_Generated automatically from the model's output probabilities and Grad-CAM map — a "
        "screening aid, not a clinical diagnosis. Always confirm with a qualified ophthalmologist._"
    )
    return "\n\n".join(lines)


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("👁️ NetraSeva — Diabetic Retinopathy Detection")
st.caption("EfficientNet-based classifier trained on the APTOS 2019 dataset, with Grad-CAM explainability")

try:
    with st.spinner("Loading model..."):
        bundle = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

model = bundle["model"]
resolution = bundle["resolution"]
detected_arch = bundle["arch"]

with st.expander(f"⚙️ Backbone: `{detected_arch}` · input {resolution}px — diagnostics"):
    st.json(bundle["diagnostics"])

with st.expander("🔬 Class order (fixes 'Mild shown as Severe'-type mislabeling)"):
    st.markdown(
        "If a class is being predicted correctly by the model but **shown under the wrong "
        "name**, it's because the index→label order below doesn't match how the model was "
        "trained. Check your training notebook for `print(train_dataset.classes)` (if you used "
        "`ImageFolder`) or confirm you used the raw `diagnosis` column order, then pick the "
        "matching preset here. This changes only the *display labels*, not the model's math — "
        "it's safe to try both and compare against images with a known true label."
    )
    order_choice = st.selectbox("Active class order", list(CLASS_ORDER_PRESETS.keys()),
                                 index=list(CLASS_ORDER_PRESETS.keys()).index(ACTIVE_ORDER_KEY))
class_names = CLASS_ORDER_PRESETS[order_choice]

uploaded_file = st.file_uploader("Retina image upload karo", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded_file is not None:
    raw_image = Image.open(io.BytesIO(uploaded_file.read()))
    display_image, input_tensor = preprocess_image(raw_image, resolution)

    left, right = st.columns([1, 1])
    with left:
        st.image(display_image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict", type="primary", use_container_width=True):
        with st.spinner("Running inference (with test-time augmentation)..."):
            probs = predict_with_tta(model, input_tensor)
            pred = int(torch.argmax(probs).item())
            conf = float(probs[pred].item()) * 100
            probs_np = probs.detach().numpy()

            target_layer = get_target_layer(model)
            gradcam_ok = False
            resized_heatmap = None
            overlay = None
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

        if probs_np.std() < 0.02:
            st.warning(
                "⚠️ All class probabilities are nearly identical — the loaded weights may still be "
                "(partially) untrained. Check the diagnostics panel above."
            )

        with right:
            if gradcam_ok:
                st.image(overlay, caption="Grad-CAM Explanation", use_container_width=True)
            else:
                st.warning(f"Grad-CAM could not be generated: {gradcam_error}")

        # ---- Result card ----
        color = CLASS_COLORS[pred]
        pred_name = class_names[pred]
        st.markdown(
            f"""
            <div class="result-card" style="border-left: 5px solid {color};">
                <span class="badge" style="background:{color};">{CLASS_ICONS[pred]} {pred_name}</span>
                <div class="conf-sub">Model confidence: <b>{conf:.2f}%</b> (TTA-averaged over original + flipped view)
                &nbsp;·&nbsp; raw model index: <b>{pred}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---- Probability chart ----
        st.write("#### Class probabilities")
        probs_df = pd.DataFrame(
            {"Class": [f"{i}: {n}" for i, n in enumerate(class_names)], "Probability (%)": probs_np * 100}
        ).set_index("Class")
        st.bar_chart(probs_df, use_container_width=True)

        ranked = sorted(range(len(class_names)), key=lambda i: probs_np[i], reverse=True)
        cols = st.columns(len(class_names))
        for i in range(len(class_names)):
            with cols[i]:
                st.metric(f"{CLASS_ICONS[i]} {class_names[i]} (idx {i})", f"{probs_np[i]*100:.1f}%")

        # ---- Recommendation ---- (driven by the LABEL, not the raw index,
        # so it stays correct regardless of which order preset is active)
        st.write("#### Recommendation")
        if pred_name == "No DR":
            st.success("Healthy — No signs of Diabetic Retinopathy detected.")
        elif pred_name == "Mild":
            st.warning("Mild DR — Early stage. Please monitor and get periodic checkups.")
        elif pred_name == "Moderate":
            st.warning("Moderate DR — Please consult an ophthalmologist.")
        else:
            st.error(f"{pred_name} — Consult a doctor immediately.")

        mild_idx = class_names.index("Mild")
        if ranked[0] == mild_idx or (
            ranked[1] == mild_idx and (probs_np[ranked[0]] - probs_np[mild_idx]) * 100 < 20
        ):
            st.info(
                "ℹ️ Mild DR is the hardest class for this model to separate from 'No DR' and "
                "'Moderate' (it's the rarest, most subtle class in the training data). If this "
                "case looks borderline, weight the Grad-CAM region and probability spread below "
                "more heavily than the single top label."
            )

        # ---- AI Analysis ----
        st.write("#### 🧠 AI Analysis")
        if gradcam_ok and resized_heatmap is not None:
            st.markdown(generate_ai_explanation(pred, probs_np, resized_heatmap, class_names))
        else:
            st.info("AI analysis requires Grad-CAM output, which failed for this image.")

        st.caption(
            "This tool is a screening aid, not a diagnostic device. "
            "Always confirm results with a qualified ophthalmologist."
        )
else:
    st.info("Upload a fundus image to get a prediction.")
