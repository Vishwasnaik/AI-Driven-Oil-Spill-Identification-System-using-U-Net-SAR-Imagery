import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os

# Add the current directory to sys.path to resolve imports regardless of where script is run from
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from unet_model import UNet
from data_loader import SyntheticOilSpillDataset
import time

def train_model(epochs=5, batch_size=4, lr=1e-3, save_path="unet_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Dataset and DataLoader
    print("Initializing Synthetic Dataset...")
    dataset = SyntheticOilSpillDataset(size=500) # 500 images
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Initialize Model
    model = UNet(n_channels=1, n_classes=1).to(device)

    # 3. Loss and Optimizer
    # BCEWithLogitsLoss combines Sigmoid + BCELoss, more stable
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4. Training Loop
    print("Starting Training...")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {epoch_loss/len(dataloader):.4f}")

    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")

    # 5. Save Model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    # Create the directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    train_model(save_path="models/unet_model.pth")
