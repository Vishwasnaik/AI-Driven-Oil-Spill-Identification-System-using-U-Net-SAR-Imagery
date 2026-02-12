# 🌊 AI-Driven Oil Spill Identification System

An end-to-end AI project for detecting oil spills from Synthetic Aperture Radar (SAR) imagery using U-Net and Streamlit.

## 📂 Project Structure
```
oil_spill_monitoring/
├── src/
│   ├── unet_model.py    # U-Net Architecture (PyTorch)
│   ├── data_loader.py   # Synthetic Data Generator & SAR Loader
│   ├── preprocessing.py # Speckle filtering & dB conversion
│   └── train.py         # Training Loop
├── app.py               # Streamlit Dashboard
├── requirements.txt     # Dependencies
└── README.md            # You are here!
```

## 🚀 Getting Started

### 1. Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Train the Model
Before running the dashboard, you need a trained model. You can train one on synthetic data instantly:
```bash
python src/train.py
```
*This will create `models/unet_model.pth`.*

### 3. Launch the Dashboard
Run the Streamlit app to visualize the results:
```bash
streamlit run app.py
```

## 🧠 How It Works
1.  **Data Generation:** The system generates synthetic SAR images (noisy ocean + dark oil patches) if real data is not provided.
2.  **Model:** A standard U-Net segments the image into "Oil" and "Ocean".
3.  **Visualization:** The results are overlaid on the original image and mapped using Folium.

## 📝 Next Steps
-   Replace `SyntheticOilSpillDataset` in `src/data_loader.py` with real Sentinel-1 data loading logic.
-   Tune the `simple_speckle_filter` in `src/preprocessing.py`.
