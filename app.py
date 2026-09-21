import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import gdown

st.set_page_config(page_title="Retinopathy Detector", layout="centered")
st.title("Diabetic Retinopathy Detector")
st.write("Retina image upload karo, AI severity batayega.")

MODEL_PATH = 'best_model.pth'
FILE_ID = '1t0FecrXJeVAAqaqpmmpcP72XIhlPpBg4'

# Model agar nahi hai to Drive se download karo
if not os.path.exists(MODEL_PATH):
    st.info("Pehli baar model download ho raha hai... 41MB, 1 min lagega")
    url = f'https://drive.google.com/uc?id={FILE_ID}'
    gdown.download(url, MODEL_PATH, quiet=False)
    st.success("Model download ho gaya!")

@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(1280, 5)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()
classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

uploaded = st.file_uploader("Image upload karo", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Retina Image", use_container_width=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        result = classes[predicted.item()]
        conf = torch.softmax(output, dim=1)[0][predicted.item()] * 100

    st.success(f"**Prediction: {result}**")
    st.metric("Confidence", f"{conf:.2f}%")
