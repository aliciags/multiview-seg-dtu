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
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from dataloader import get_dataloader, walk_through_dir
from sklearn.metrics import precision_score, recall_score, f1_score
from models import SimpleSegmentationModel, SegmentationModel, pretrained_UNet

def set_seed(seed=111):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    random.seed(seed)
    np.random.seed(seed)

# Training loop
def train_model_old(model, train_loader, criterion, optimizer, num_epochs):
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

# Funció per calcular el Dice Score
def calculate_dice_score(outputs, masks):
    outputs = (outputs > 0.5).float()  # Llindar per prediccions binàries
    intersection = (outputs * masks).sum()
    union = outputs.sum() + masks.sum()
    dice_score = (2 * intersection) / (union + 1e-7)  # Evita divisió per zero
    return dice_score.item()

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_losses = []
    val_losses = []
    train_dice_scores = []
    val_dice_scores = []

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        train_loss = 0
        train_dice_score = 0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss and Dice Score
            train_loss += loss.item()
            train_dice_score += calculate_dice_score(outputs, masks)

        # calculat metricas del training
        train_loss /= len(train_loader)
        train_dice_score /= len(train_loader)
        train_losses.append(train_loss)
        train_dice_scores.append(train_dice_score)

        # Avaluació en el set de validació
        model.eval()  # Set model to evaluation mode
        val_loss = 0
        val_dice_score = 0

        with torch.no_grad(): #aqui validation stepppp para tener el test sin tocar
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item()
                val_dice_score += calculate_dice_score(outputs, masks)

        # Calcula mètriques de validació
        val_loss /= len(val_loader)
        val_dice_score /= len(val_loader)
        val_losses.append(val_loss)
        val_dice_scores.append(val_dice_score)

        # Imprimeix mètriques
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Dice Score: {train_dice_score:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Dice Score: {val_dice_score:.4f}")

    return train_losses, val_losses, train_dice_scores, val_dice_scores

# Evaluate the model
def evaluate_model(model, test_dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    all_preds = []  # List to store all predictions
    all_labels = []  # List to store all ground truth labels
    
    with torch.no_grad():  # Disable gradient computation
        for images, masks in test_dataloader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            # Convert outputs to binary predictions (threshold at 0.5 for binary classification)
            binary_outputs = (outputs > 0.5).float()

            # Flatten the outputs and masks for evaluation
            binary_outputs_np = binary_outputs.cpu().numpy().flatten()
            masks_np = masks.cpu().numpy().flatten()

            # Collect predictions and true labels for metrics calculation
            all_preds.extend(binary_outputs_np)
            all_labels.extend(masks_np)

    # Calculate average loss
    avg_loss = total_loss / len(test_dataloader)
    
    # Ensure that labels and predictions are both binary (0 or 1)
    all_preds = [int(x) for x in all_preds]  # Convert predictions to integer binary
    all_labels = [int(x) for x in all_labels]  # Convert labels to integer binary

    # Compute Precision, Recall, and F1 score
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    # Print results
    print(f"Average Loss on Test Set: {avg_loss:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script arguments")
    parser.add_argument(
        "--num_epochs", 
        type=int, 
        required=True, 
        help="Number of epochs to train the model"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        required=True, 
        help="Name of the model to train (Simple/UNet/Pretrained)"
    )
    args = parser.parse_args()
    
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
    
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(image_dirs, mask_dir, data_transform, mask_transform, display_sample=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model, loss function, and optimizer
    model_name = args.model_name
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
    #train_model(model, train_dataloader, criterion, optimizer, num_epochs=25)
    # Entrena el model
    train_losses, val_losses, train_dice_scores, val_dice_scores = train_model(
        model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=args.num_epochs,
    )
    
    # Save the model
    current_directory = os.getcwd()
    model_name_saved = "segmentation_model_" + model_name + ".pth"
    model_save_path = os.path.join(current_directory, model_name_saved)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved successfully to {model_save_path}")

    # Pèrdua (Loss)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(f"{model_name}_training_validation_loss.png")  # Save with model name
    plt.close()  # Close the plot to free memory

    # Dice Score
    plt.figure(figsize=(10, 5))
    plt.plot(train_dice_scores, label="Train Dice Score")
    plt.plot(val_dice_scores, label="Validation Dice Score")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.legend()
    plt.title("Training and Validation Dice Score")
    plt.savefig(f"{model_name}_training_validation_dice_score.png")  # Save with model name
    plt.close()  # Close the plot to free memory

    evaluate_model(model, test_dataloader, criterion, device)
