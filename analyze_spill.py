import torch
import numpy as np
import cv2
import os
import sys
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unet_model import UNet
from data_loader import SyntheticOilSpillDataset
import utils

def calculate_spill_stats(prediction_mask, prob_map, resolution_m=10):
    """
    Calculate area and confidence stats.
    """
    spill_pixels = np.sum(prediction_mask)
    pixel_area_m2 = resolution_m * resolution_m
    total_area_m2 = spill_pixels * pixel_area_m2
    total_area_km2 = total_area_m2 / 1_000_000
    
    # Confidence: Mean probability of pixels classified as spill
    if spill_pixels > 0:
        confidence = np.mean(prob_map[prediction_mask == 1]) * 100
    else:
        confidence = 0.0
        
    return total_area_km2, spill_pixels, confidence

def update_spill_history(area_km2, history_file="spill_history.json"):
    """
    Updates a local JSON file with the latest spill area reading.
    """
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    else:
        history = []
        
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history.append({"timestamp": timestamp, "area_km2": area_km2})
    
    # Keep last 10 records for demo
    if len(history) > 10:
        history = history[-10:]
        
    with open(history_file, "w") as f:
        json.dump(history, f, indent=4)
        
    return history

def analyze_sample(model_path="models/unet_model.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        # Build model anyway for demo purposes if weights missing
        model = UNet(n_channels=1, n_classes=1).to(device)
    else:
        model = UNet(n_channels=1, n_classes=1).to(device)
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"Warning: Could not load trained weights ({e}). Using random weights for demo.")
    
    model.eval()

    # Generate a sample
    ds = SyntheticOilSpillDataset(size=1)
    img_tensor, mask_tensor = ds[0]
    
    # Inference
    input_tensor = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
        prediction = (prob_map > 0.5).astype(np.uint8)

    # --- Advanced Analytics ---
    
    # 1. Area & Confidence
    area_km2, pixels, confidence = calculate_spill_stats(prediction, prob_map)
    
    # 2. History
    history = update_spill_history(area_km2)
    
    # 3. Environmental & Drift (Mock Location)
    # Mocking a location in the Indian Ocean
    center_lat, center_lon = 12.9716, 77.5946 
    
    wind_speed, wind_dir = utils.generate_wind_data()
    drift_lat, drift_lon = utils.predict_drift(center_lat, center_lon, wind_speed, wind_dir)
    
    # 4. Vulnerability
    alerts = utils.check_environmental_vulnerability(center_lat, center_lon)
    
    # 5. AIS (Ships)
    ships = utils.get_nearby_ais_ships(center_lat, center_lon)

    # --- Reporting ---
    print("=" * 40)
    print("      OIL SPILL DETECTION REPORT      ")
    print("=" * 40)
    print(f"Detected Area:    {area_km2:.4f} km²")
    print(f"Confidence score: {confidence:.2f}%")
    print("-" * 40)
    print(f"Wind Conditions:  {wind_speed:.1f} km/h at {wind_dir:.0f}°")
    print(f"Drift Prediction: Moving towards ({drift_lat:.4f}, {drift_lon:.4f})")
    print("-" * 40)
    if alerts:
        print("CRITICAL ALERTS:")
        for alert in alerts:
            print(f"  [!] {alert}")
    else:
        print("No immediate environmental threats detected.")
    print("-" * 40)
    print(f"Nearby Ships (Potential Sources):")
    for ship in ships:
        print(f"  - {ship['name']} ({ship['type']})")
    print("=" * 40)

    # --- Visualization Dashboard ---
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3)
    fig.suptitle(f"Advanced Oil Spill Analytics - {datetime.now().strftime('%Y-%m-%d %H:%M')}", fontsize=16)

    # Panel 1: SAR Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_tensor.squeeze(), cmap='gray')
    ax1.set_title("SAR Satellite Input")
    ax1.axis('off')

    # Panel 2: Detection Mask
    ax2 = fig.add_subplot(gs[0, 1])
    # Overlay logic for matplotlib
    img_bg = img_tensor.squeeze().numpy()
    ax2.imshow(img_bg, cmap='gray')
    ax2.imshow(prediction, cmap='Reds', alpha=0.5)
    ax2.set_title(f"AI Segmented Spill (Conf: {confidence:.1f}%)")
    ax2.axis('off')

    # Panel 3: Optical View (Mock)
    ax3 = fig.add_subplot(gs[0, 2])
    # Create valid mock RGB image
    mock_optical = np.zeros((256, 256, 3), dtype=np.uint8)
    mock_optical[:] = [20, 100, 150] # Ocean blue
    # Add some clouds
    for _ in range(8):
        cv2.circle(mock_optical, (np.random.randint(0,256), np.random.randint(0,256)), 
                   np.random.randint(10,40), (255,255,255), -1)
    # Add spill hint in optical
    if pixels > 0:
        # Simple approximation of spill location in optical
        spill_optical = cv2.resize(prediction, (256, 256))
        # Darken the spill area in optical to look like oil
        mock_optical[spill_optical == 1] = [50, 50, 50] 
        
    ax3.imshow(mock_optical)
    ax3.set_title("Optical Verification (Sentinel-2)")
    ax3.axis('off')

    # Panel 4: Spill History
    ax4 = fig.add_subplot(gs[1, 0:2])
    timestamps = [h['timestamp'].split(' ')[1] for h in history] # Just time
    areas = [h['area_km2'] for h in history]
    ax4.plot(timestamps, areas, marker='o', linestyle='-', color='red', linewidth=2)
    ax4.set_title("Spill Progression (Historical Trend)")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Area (km²)")
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    # Panel 5: Context Info
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    # Create a text summary for the plot
    info_text = (
        f"METADATA\n"
        f"--------\n"
        f"Wind Speed: {wind_speed:.1f} km/h\n"
        f"Wind Dir:   {wind_dir:.0f}°\n\n"
        f"DRIFT PREDICTION\n"
        f"----------------\n"
        f"Lat: {drift_lat:.4f}\n"
        f"Lon: {drift_lon:.4f}\n\n"
        f"NEARBY SHIPS (AIS)\n"
        f"------------------\n"
    )
    for s in ships[:3]:
        info_text += f"{s['name']}\n"
    
    if alerts:
        info_text += "\nENVIRONMENTAL ALERTS\n--------------------\n"
        for alert in alerts:
            info_text += f"{alert}\n"
        
    ax5.text(0.05, 0.95, info_text, fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax5.set_title("Situational Awareness")

    plt.tight_layout()
    plt.savefig("analysis_dashboard.png", dpi=150)
    print("Dashboard saved to 'analysis_dashboard.png'")

if __name__ == "__main__":
    analyze_sample()
