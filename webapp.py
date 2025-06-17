import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np


model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 6) 
model.load_state_dict(torch.load('best_model_ResNet18.pt')) 
model.eval()

# Define the transformation for input image (resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Class labels for prediction
class_labels = ['building', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Streamlit UI components
st.title("Image Classification with ResNet18")
st.write("Upload an image to classify it into one of the six classes.")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Transform the image for the model
    image_tensor = transform(image).unsqueeze(0)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_class = torch.max(outputs, 1)
    
    predicted_label = class_labels[predicted_class.item()]
    
    st.write(f"Predicted Class: {predicted_label}")
