
import torch
from src.unet_model import UNet

def test_unet():
    model = UNet(n_channels=3, n_classes=1)
    x = torch.randn(1, 3, 572, 572)
    try:
        y = model(x)
        print("Forward pass successful")
        print("Output shape:", y.shape)
    except Exception as e:
        print("Forward pass failed")
        print(e)
        raise e

if __name__ == "__main__":
    test_unet()
