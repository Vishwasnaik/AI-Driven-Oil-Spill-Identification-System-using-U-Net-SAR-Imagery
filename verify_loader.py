
import numpy as np
import cv2
from src.data_loader import SyntheticOilSpillDataset

def test_loader():
    try:
        ds = SyntheticOilSpillDataset(size=5)
        img, mask = ds[0]
        print("Data loader simple access successful")
        print("Image shape:", img.shape)
        print("Mask shape:", mask.shape)
        
        # Test cv2 drawing on float32
        test_mask = np.zeros((256, 256), dtype=np.float32)
        cv2.ellipse(test_mask, (128, 128), (50, 30), 0, 0, 360, 1.0, -1)
        print("cv2.ellipse on float32 successful")
        
    except Exception as e:
        print("Data loader failed")
        print(e)

if __name__ == "__main__":
    test_loader()
