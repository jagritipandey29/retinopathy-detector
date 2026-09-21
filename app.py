import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.title("Retinopathy Detector")
st.write("Image upload karo")

# 1. Model structure
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(1280, 5)

# 2. Trained weights load karo - YEH SABSE ZARURI HAI
# Tera jo.pth file hai Kaggle wala, uska naam yaha daal
# Jaise 'best_model.pth' ya 'retinopathy_efficientnet.pth'
try:
    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    st.success("Model load ho gaya!")
except:
    st.error("Model file nahi mila! GitHub me.pth file add karo")

model.eval()

# Classes - apne hisab se naam change kar lena
classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

uploaded = st.file_uploader("Image upload karo", type=["jpg","png","jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # 3. Prediction logic
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    img_tensor = transform(img).unsqueeze(0) # batch bana diya

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        result = classes[predicted.item()]
        confidence = torch.softmax(output, dim=1)[0][predicted.item()] * 100

    st.success(f"**Prediction: {result}**")
    st.info(f"Confidence: {confidence:.2f}%")
