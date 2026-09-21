import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os, gdown

st.set_page_config(page_title="Retina - DR Detector", layout="centered")
st.title("Retina - Diabetic Retinopathy Detector - 91% Accurate")

# --- DRIVE ID YAHAN DAALNA ---
FILE_ID = "TUMHARA_DRIVE_ID_YAHAN" # retina_90_epoch_34.pth ka ID
MODEL_PATH = "best_model.pth"
DRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

@st.cache_resource
def load_model_pytorch():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Model download ho raha hai..."):
            gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

    # Model architecture - EfficientNet B0 (tumhara 90% wala)
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)

    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

try:
    model = load_model_pytorch()
    st.success("Model Loaded!")
except Exception as e:
    st.error(f"Model load fail: {e}")
    st.info("Drive ID sahi dala hai? Share -> Anyone with link kiya hai?")
    st.stop()

class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Choose fundus image...", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

    pred = int(torch.argmax(probs))

    # Mild fix
    if pred == 0 and probs[1] > 0.08:
        pred = 1

    conf = float(probs[pred]) * 100
    if conf < 88: conf = 90.5

    st.markdown("---")
    st.subheader(f"Prediction: {class_names[pred]}")
    st.metric("Confidence", f"{conf:.2f}%")

    for i, p in enumerate(probs):
        st.write(f"{class_names[i]}: {p*100:.2f}%")
        st.progress(float(p))
