import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.title("Retinopathy Detector")
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(1280, 5)
model.eval()

uploaded = st.file_uploader("Image upload karo", type=["jpg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img)
    st.success("Model yaha predict karega")
