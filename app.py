import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import gdown

st.set_page_config(page_title="Retina Detector - 81.99% Accurate", layout="centered")
st.title("Retina - Diabetic Retinopathy Detector - 81.99% Accurate")

MODEL_PATH = 'best_model.pth'
FILE_ID = '1t0FecrXJeVAAqaqpmmpcP72XIhlPpBg4'

if not os.path.exists(MODEL_PATH):
    st.info("Model download ho raha hai...")
    url = f'https://drive.google.com/uc?id={FILE_ID}'
    gdown.download(url, MODEL_PATH, quiet=False)

@st.cache_resource
def load_model():
    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(1536, 5)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()
classes = ['No DR (0)', 'Mild (1)', 'Moderate (2)', 'Severe (3)', 'Proliferative DR (4)']

uploaded = st.file_uploader("Img upload karo", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0] * 100
        _, predicted = torch.max(output, 1)
        result_idx = predicted.item()

    st.subheader(f"Predicted: {classes[result_idx]}")

    st.write("### Confidence")
    for i, cls in enumerate(classes):
        st.write(f"{cls}")
        st.progress(int(probs[i].item()))
        st.write(f"{probs[i].item():.2f}%")

    st.success(f"Result: Predicted: {classes[result_idx]}")
