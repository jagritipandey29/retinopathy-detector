import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import os
import gdown
import plotly.express as px
import plotly.graph_objects as go
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="RetinaAI - DR Screening System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR DYNAMIC THEME SUPPORT (LIGHT & DARK BOTH) ---
st.markdown("""
    <style>
    /* Metric Cards - Theme adaptive background & text */
    [data-testid="stMetric"] {
        background-color: var(--background-secondary-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Make metric label and value adapt to theme text color */
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: var(--text-color) !important;
    }

    /* Dynamic Alert Banners */
    .status-card-danger {
        background-color: rgba(255, 75, 75, 0.15);
        border-left: 6px solid #ff4b4b;
        padding: 15px;
        border-radius: 8px;
        color: var(--text-color);
        font-weight: bold;
    }
    .status-card-success {
        background-color: rgba(0, 200, 83, 0.15);
        border-left: 6px solid #00c853;
        padding: 15px;
        border-radius: 8px;
        color: var(--text-color);
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- MODEL LOADING WITH CACHE ---
MODEL_ID = "1f4mtlffwi9omaJKERX-KRY4hrk86B5Uz"
MODEL_PATH = "best_qwk_final.pth"

@st.cache_resource
def load_dr_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", MODEL_PATH, quiet=False)
    model = models.efficientnet_b4(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

model = load_dr_model()

# --- HELPER FUNCTIONS ---
def process_quality_and_clahe(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    
    if blur_score < 50 or brightness < 15 or brightness > 235:
        return None, False, f"Image quality too low (Blur Score: {blur_score:.1f}). Please re-capture."
    
    denoised = cv2.fastNlMeansDenoisingColored(img_np, None, 5, 5, 7, 21)
    green = denoised[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_green = clahe.apply(green)
    
    enhanced_img = denoised.copy()
    enhanced_img[:, :, 1] = enhanced_green
    return enhanced_img, True, "Quality Check Passed & CLAHE Enhancement Applied."

def segment_lesions(img_np):
    green = img_np[:, :, 1]
    kernel_v = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    top_v = cv2.morphologyEx(green, cv2.MORPH_TOPHAT, kernel_v)
    _, vessels = cv2.threshold(top_v, 15, 255, cv2.THRESH_BINARY)
    
    kernel_m = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    top_m = cv2.morphologyEx(green, cv2.MORPH_TOPHAT, kernel_m)
    _, ma = cv2.threshold(top_m, 20, 255, cv2.THRESH_BINARY)
    ma = cv2.bitwise_and(ma, cv2.bitwise_not(vessels))
    return vessels, ma

def generate_gradcam(model, img_tensor, img_np_resized):
    try:
        target_layers = [model.features[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=img_tensor, targets=None)[0, :]
        rgb_float = img_np_resized.astype(np.float32) / 255.0
        return show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    except Exception as e:
        gray = cv2.cvtColor(img_np_resized, cv2.COLOR_RGB2GRAY)
        heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(img_np_resized, 0.6, heatmap, 0.4, 0)

# --- DYNAMIC URL PARAMETERS FETCH (For MATLAB / External Integration) ---
query_params = st.query_params

default_patient_id = query_params.get("patient_id", "PT-88392")
try:
    default_age = int(query_params.get("age", 54))
except ValueError:
    default_age = 54
default_eye = query_params.get("eye", "Right Eye (OD)")
logged_in_user = query_params.get("user_name", "Doctor / Operator")

eye_index = 1 if "left" in default_eye.lower() or "os" in default_eye.lower() else 0

# --- SIDEBAR: CLINICAL CONTROLS & PATIENT INFO ---
st.sidebar.markdown(f"👨‍⚕️ **Active User:** {logged_in_user}")
st.sidebar.markdown("---")
st.sidebar.title("🩺 Patient Details")

patient_id = st.sidebar.text_input("Patient ID", value=default_patient_id)
patient_age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=default_age)
eye_side = st.sidebar.selectbox("Eye Side", ["Right Eye (OD)", "Left Eye (OS)"], index=eye_index)

st.sidebar.markdown("---")
st.sidebar.caption("🔒 Model: EfficientNet-B4 (Trained with QWK Loss)")

# --- MAIN HEADER ---
st.title("👁️ Retinal Diabetic Retinopathy Screening System")
st.write("Automated AI Diagnostics, Lesion Segmentation & Explainability Visuals")

uploaded_file = st.file_uploader("Upload Retinal Fundus Scan Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    raw_img = Image.open(uploaded_file).convert("RGB")
    raw_np = np.array(raw_img)
    
    enhanced_np, is_pass, quality_msg = process_quality_and_clahe(raw_np)
    
    if is_pass:
        img_resized = cv2.resize(enhanced_np, (380, 380))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img_resized).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1).numpy()[0]
            pred_class = int(np.argmax(probs))
            conf = float(probs[pred_class] * 100)
            
        labels = [
            "No DR (Normal)",
            "Mild DR",
            "Moderate DR",
            "Severe DR",
            "Proliferative DR"
        ]
        
        is_referable = pred_class >= 2
        
        # --- HEADER KPI METRICS ---
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Predicted Diagnosis", labels[pred_class])
        col_m2.metric("Confidence Score", f"{conf:.1f}%")
        col_m3.metric("Referable DR Status", "YES" if is_referable else "NO")
        col_m4.metric("Image Quality Check", "Passed (CLAHE)")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- ALERT BANNER ---
        if is_referable:
            st.markdown(f'<div class="status-card-danger">⚠️ REFERABLE DR DETECTED: High likelihood of Diabetic Retinopathy ({labels[pred_class]}). Immediate Ophthalmologist referral recommended.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="status-card-success">✅ NO REFERABLE DR: Scan indicates low risk ({labels[pred_class]}). Routine annual follow-up recommended.</div>', unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)

        # --- TABS FOR ORGANIZED PRESENTATION ---
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Diagnosis & Probabilities",
            "🔍 Structural Lesion Masking",
            "🧠 AI Grad-CAM Heatmap",
            "📈 Telemedicine Load Simulator"
        ])
        
        # TAB 1: DIAGNOSIS & PLOTLY CHARTS
        with tab1:
            col_img, col_chart = st.columns([1, 1])
            with col_img:
                st.subheader("Enhanced Fundus Image")
                st.image(enhanced_np, use_container_width=True, caption=f"Patient: {patient_id} ({eye_side})")
                
            with col_chart:
                st.subheader("Stage-wise Probability Distribution")
                fig_bar = px.bar(
                    x=probs * 100,
                    y=labels,
                    orientation='h',
                    labels={'x': 'Probability (%)', 'y': 'DR Severity Stage'},
                    color=probs * 100,
                    color_continuous_scale='Reds' if is_referable else 'Greens'
                )
                fig_bar.update_layout(showlegend=False, height=350, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_bar, use_container_width=True)

        # TAB 2: LESION SEGMENTATION
        with tab2:
            st.subheader("Extracted Vascular & Lesion Features")
            vessels, microaneurysms = segment_lesions(img_resized)
            
            c_v1, c_v2, c_v3 = st.columns(3)
            c_v1.image(img_resized, caption="Original Fundus", use_container_width=True)
            c_v2.image(vessels, caption="Vessel Structure Mask", use_container_width=True)
            c_v3.image(microaneurysms, caption="Detected Microaneurysms", use_container_width=True)

        # TAB 3: GRAD-CAM HEATMAP
        with tab3:
            st.subheader("Model Attention Region (Grad-CAM)")
            gradcam_result = generate_gradcam(model, img_tensor, img_resized)
            
            c_g1, c_g2 = st.columns(2)
            c_g1.image(img_resized, caption="Preprocessed Image", use_container_width=True)
            c_g2.image(gradcam_result, caption="AI Focus / Heatmap Highlights", use_container_width=True)

        # TAB 4: TELEMEDICINE SIMULATOR
        with tab4:
            st.subheader("Telemedicine Clinic Program Simulation")
            annual_p = st.slider("Target Annual Patient Volume", 10000, 200000, 50000, step=10000)
            
            daily_scans = int(annual_p / 365)
            referral_cases = int(daily_scans * 0.18)
            doc_hours = (referral_cases * 3) / 60
            
            sim_col1, sim_col2, sim_col3 = st.columns(3)
            sim_col1.metric("Daily Patient Scans", f"{daily_scans} / day")
            sim_col2.metric("Expected Referral Cases", f"{referral_cases} / day")
            sim_col3.metric("Doctor Review Time", f"{doc_hours:.1f} Hours / day")
            
    else:
        st.error(quality_msg)
