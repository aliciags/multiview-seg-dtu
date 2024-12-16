import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np

from models import SegmentationModel
from dataloader import get_dataloader, walk_through_dir
from pathlib import Path
import random

# check if gpu available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data_path = "/zhome/70/5/14854/nobackup/deeplearningf24/forcebiology/data/"
walk_through_dir(Path(data_path))

image_dirs = [data_path + 'brightfield/Alexa488_Fibroblasts_well1_50locations',
              data_path + 'brightfield/Alexa488_Fibroblasts_well2_200locations',
              data_path + 'brightfield/Alexa488_Fibroblasts_well3_200locations',
              data_path + 'brightfield/Alexa488_Fibroblasts_well4_225locations',
              data_path + 'brightfield/Alexa488_Fibroblasts_well5_225locations',
              data_path + 'brightfield/Alexa488_Fibroblasts_well6_135locations',
              data_path + 'brightfield/Alexa488_Fibroblasts_well7_135locations']
mask_dir = data_path + 'masks'

def set_seed(seed=111):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    random.seed(seed)
    np.random.seed(seed)

set_seed()

data_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
mask_transform = data_transform

train_dataloader, test_dataloader = get_dataloader(image_dirs, mask_dir, data_transform, mask_transform, display_sample=False)

# K-Fold Cross-Validation Function
def cross_validate_with_dataset(
    model_class,
    dataset,
    k=5,
    num_epochs=50,
    batch_size=32,
    learning_rate=1e-3,
    device='cuda',
):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)))):
        print(f"Fold {fold+1}/{k}")
        
        # Split dataset into training and validation subsets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        print('subsets splitted')

        # Initialize DataLoaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        print('Dataloaded')
        
        # Initialize model, loss, optimizer
        model = model_class().to(device)
        criterion = nn.BCELoss()  # For binary segmentation masks
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        print('initialize model')
        
        train_losses, val_losses = [], []

        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Store results for this fold
        fold_results.append({'train_losses': train_losses, 'val_losses': val_losses})

    return fold_results

# Plot losses for analysis
def plot_losses(fold_results):
    for fold, result in enumerate(fold_results):
        plt.plot(result['train_losses'], label=f"Train Loss (Fold {fold+1})")
        plt.plot(result['val_losses'], label=f"Val Loss (Fold {fold+1})", linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("Train vs. Validation Loss Across Folds")
    plt.show()

train_dataset = train_dataloader.dataset

fold_results = cross_validate_with_dataset(
    model_class=SegmentationModel,
    dataset=train_dataset,
    k=5,
    num_epochs=20,
    batch_size=32,
    learning_rate=1e-3,
    device='cuda' if torch.cuda.is_available() else 'cpu',
)

plot_losses(fold_results)