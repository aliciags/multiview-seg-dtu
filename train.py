import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
from pathlib import Path
from dataloader import get_dataloader, walk_through_dir
from models import SimpleSegmentationModel, SegmentationModel, pretrained_UNet

def set_seed(seed=111):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    random.seed(seed)
    np.random.seed(seed)

# Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        epoch_loss = 0
        start_time = time.time()  # Record the start time of the epoch

        for images, masks in train_loader:
            # Move images and masks to the device (GPU if available)
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate the loss for this batch
            epoch_loss += loss.item()

        # Calculate epoch duration
        epoch_duration = time.time() - start_time

        # Average loss for this epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}, Time: {epoch_duration:.2f} seconds")
            

if __name__ == "__main__":
    set_seed()

    data_path = "/zhome/70/5/14854/nobackup/deeplearningf24/forcebiology/data/" 
    image_dirs = [data_path + 'brightfield/Alexa488_Fibroblasts_well1_50locations',
                  data_path + 'brightfield/Alexa488_Fibroblasts_well2_200locations',
                  data_path + 'brightfield/Alexa488_Fibroblasts_well3_200locations',
                  data_path + 'brightfield/Alexa488_Fibroblasts_well4_225locations',
                  data_path + 'brightfield/Alexa488_Fibroblasts_well5_225locations',
                  data_path + 'brightfield/Alexa488_Fibroblasts_well6_135locations',
                  data_path + 'brightfield/Alexa488_Fibroblasts_well7_135locations']
    mask_dir = data_path + 'masks'
    
    data_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    mask_transform = data_transform
    
    train_dataloader, test_dataloader = get_dataloader(image_dirs, mask_dir, data_transform, mask_transform, display_sample=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model, loss function, and optimizer
    model_name = "Pretrained"
    if model_name == "Simple":
        model = SimpleSegmentationModel().to(device)
        criterion = nn.BCELoss()
    elif model_name == "UNet":
        model = SegmentationModel().to(device)
        criterion = nn.BCELoss()
    elif model_name == "Pretrained":
        model = pretrained_UNet().to(device)
        criterion = nn.BCEWithLogitsLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model with the training dataloader
    train_model(model, train_dataloader, criterion, optimizer, num_epochs=25)

    # Save the model
    current_directory = os.getcwd()
    model_name_saved = "segmentation_model_" + model_name + ".pth"
    model_save_path = os.path.join(current_directory, model_name_saved)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved successfully to {model_save_path}")
