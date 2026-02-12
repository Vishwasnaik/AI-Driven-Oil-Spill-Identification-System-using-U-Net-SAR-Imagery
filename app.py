import streamlit as st
import folium
from streamlit_folium import st_folium
import torch
import numpy as np
import cv2
from PIL import Image
import os
import sys

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from unet_model import UNet
from data_loader import SyntheticOilSpillDataset
from preprocessing import normalize

# Page Config
st.set_page_config(page_title="Oil Spill Monitor", page_icon="🌊", layout="wide")

# Title and Sidebar
st.title("🌊 AI-Driven Oil Spill Identification System")
st.markdown("""
This system uses Synthetic Aperture Radar (SAR) imagery and Deep Learning (U-Net) 
to detect and map oil spills in the ocean.
""")

st.sidebar.header("Control Panel")
model_path = st.sidebar.text_input("Model Path", "models/unet_model.pth")

@st.cache_resource
def load_model(path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=1, n_classes=1)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model
    else:
        return None

# Load Model
model = load_model(model_path)
if model is None:
    st.sidebar.warning(f"Model not found at {model_path}. Please run src/train.py first!")

# Input Section
st.header("1. Data Acquisition")
option = st.radio("Choose Input Source:", ("Upload SAR Image", "Generate Synthetic Sample"))

input_image = None
ground_truth = None

if option == "Upload SAR Image":
    uploaded_file = st.file_uploader("Upload a grayscale SAR image (PNG/JPG/TIF)", type=["png", "jpg", "tif"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('L')
        input_image = np.array(image)
        # Normalize to 0-1
        input_image = input_image.astype(np.float32) / 255.0

elif option == "Generate Synthetic Sample":
    if st.button("Generate Random Sample"):
        ds = SyntheticOilSpillDataset(size=1)
        img_tensor, mask_tensor = ds[0]
        input_image = img_tensor.squeeze().numpy()
        ground_truth = mask_tensor.squeeze().numpy()
        st.success("Generated synthetic SAR image with simulated oil spill.")

# Inference Section
if input_image is not None:
    st.header("2. Analysis & Detection")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Input SAR Image")
        # Display as grayscale
        st.image(input_image, caption="Processed Input", use_container_width=True, clamp=True)

    # Run Inference
    if model:
        # Prepare tensor
        img_tensor = torch.from_numpy(input_image).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            output = model(img_tensor)
            # Sigmoid to get probability 0-1
            prob_map = torch.sigmoid(output).squeeze().numpy()
            # Threshold at 0.5
            prediction = (prob_map > 0.5).astype(np.float32)

        with col2:
            st.subheader("AI Prediction")
            st.image(prediction, caption="Predicted Oil Mask", use_container_width=True, clamp=True)
            
        with col3:
            st.subheader("Overlay")
            # Create an RGB overlay
            # normalized 0-255
            base = (input_image * 255).astype(np.uint8)
            base_rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
            
            # Red overlay for oil
            mask_overlay = (prediction * 255).astype(np.uint8)
            # Apply color map or just set red channel
            base_rgb[mask_overlay > 0] = [255, 0, 0] # Red spill
            
            st.image(base_rgb, caption="Overlay (Red = Spill)", use_container_width=True)

    else:
        st.error("Model not loaded. Cannot run inference.")

    # Mapping Section
    st.header("3. Geospatial Monitoring")
    st.markdown("Locating the detected spill on the global map.")
    
    # Mock coordinates (in real app, these come from the TIF metadata)
    lat, lon = 25.0, -80.0  # Near Florida Keys
    
    m = folium.Map(location=[lat, lon], zoom_start=8)
    
    # Add a marker
    folium.Marker(
        [lat, lon], 
        popup="Detected Oil Spill", 
        icon=folium.Icon(color="red", icon="warning-sign")
    ).add_to(m)
    
    # Add the overlay to the map (simulated context)
    # in a real app, we would project the pixel mask to lat/lon polygons
    folium.Circle(
        radius=5000,
        location=[lat, lon],
        popup="Affected Area Analysis",
        color="crimson",
        fill=True,
    ).add_to(m)

    st_folium(m, width=1200, height=500)

if st.checkbox("Show Technical Details"):
    st.write("Model Architecture: U-Net")
    st.write(f"Input Shape: {input_image.shape if input_image is not None else 'N/A'}")
