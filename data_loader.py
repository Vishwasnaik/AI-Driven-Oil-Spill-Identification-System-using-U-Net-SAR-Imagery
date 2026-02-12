import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os

class SyntheticOilSpillDataset(Dataset):
    """
    Generates synthetic SAR images with oil spill-like features for testing.
    Real SAR data is typically single-channel grayscale but often noisy (speckle).
    Oil spills appear as dark patches on a brighter ocean background.
    """
    def __init__(self, size=100, img_dim=(256, 256)):
        self.size = size
        self.img_dim = img_dim

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 1. Generate Background (Ocean with speckle noise)
        # Gamma distribution approximates SAR speckle well
        background = np.random.gamma(shape=4.0, scale=10.0, size=self.img_dim).astype(np.float32)
        
        # Normalize background to 0-1 range for typical image processing
        background = (background - background.min()) / (background.max() - background.min())

        # 2. Generate Oil Spill (Dark ellipse/irregular shapes)
        mask = np.zeros(self.img_dim, dtype=np.float32)
        
        # Random number of spills
        num_spills = np.random.randint(0, 3) 
        
        for _ in range(num_spills):
            center = (np.random.randint(0, self.img_dim[1]), np.random.randint(0, self.img_dim[0]))
            axes = (np.random.randint(10, 60), np.random.randint(10, 60))
            angle = np.random.randint(0, 180)
            cv2.ellipse(mask, center, axes, angle, 0, 360, 1, -1)

        # 3. Apply Oil Spill to Background (Oil is darker, so suppress background)
        # Darken the areas where the mask is 1
        # In SAR, smooth oil reflects less back to the sensor -> low values
        image = background * (1 - mask * 0.8) # Reduce intensity by 80% in spill areas
        
        # Add some noise to the spill area too
        noise = np.random.normal(0, 0.05, self.img_dim).astype(np.float32)
        image = image + noise
        image = np.clip(image, 0, 1)

        # Convert to Tensor (C, H, W)
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return image_tensor, mask_tensor

def save_sample_data(output_dir="data/sample"):
    """Helper to generate and save some physical files for the user to inspect."""
    os.makedirs(output_dir, exist_ok=True)
    ds = SyntheticOilSpillDataset(size=5)
    for i in range(5):
        img, mask = ds[i]
        # Convert to 0-255 uint8 for saving
        img_np = (img.squeeze().numpy() * 255).astype(np.uint8)
        mask_np = (mask.squeeze().numpy() * 255).astype(np.uint8)
        
        cv2.imwrite(f"{output_dir}/sample_{i}_img.png", img_np)
        cv2.imwrite(f"{output_dir}/sample_{i}_mask.png", mask_np)
    print(f"Saved 5 synthetic samples to {output_dir}")
