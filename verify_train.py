
from src.train import train_model
import os

def test_training():
    print("Testing training loop...")
    try:
        # Run for 1 epoch with small batch size to be quick
        train_model(epochs=1, batch_size=2, save_path="test_model.pth")
        print("Training loop successful")
        if os.path.exists("test_model.pth"):
            os.remove("test_model.pth")
    except Exception as e:
        print("Training loop failed")
        print(e)
        raise e

if __name__ == "__main__":
    test_training()
