import os
import numpy as np
import matplotlib.pyplot as plt
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
from sklearn.metrics import precision_score, recall_score, f1_score
import argparse

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

    # Compute Dice Score
    intersection = np.sum(np.array(all_preds) * np.array(all_labels))
    dice_score = (2.0 * intersection) / (np.sum(all_preds) + np.sum(all_labels) + 1e-8)

    # Compute IoU (Intersection over Union)
    union = np.sum((np.array(all_preds) + np.array(all_labels)) > 0)
    iou = intersection / (union + 1e-8)

    # Print results
    print(f"Average Loss on Test Set: {avg_loss:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Dice Score: {dice_score:.4f}")
    print(f"IoU: {iou:.4f}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training script arguments")
    parser.add_argument(
        "--model_name", 
        type=str, 
        required=True, 
        help="Choose Simple/UNet/Pretrained"
    )
    parser.add_argument(
        "--pth_model", 
        type=str, 
        required=True, 
        help="pth saved model full name (e.g. segmentation_model_UNet_train100%_channels1.pth)"
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs='+',           # Allows one or more integers (vector of channels)
        default=None,
        help="List of channel indices to use (e.g., 0 3 5). Default is None (use all channels)."
    )
    args = parser.parse_args()

    data_path = "/zhome/70/5/14854/nobackup/deeplearningf24/forcebiology/data/"
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
    
    # Get the dataloaders with optional channel selection
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(
        image_dirs, 
        mask_dir, 
        data_transform, 
        mask_transform, 
        display_sample=False, 
        train_percentage=1.0, 
        channel_indices=args.channels
    )
    
    # Initialize the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCELoss()
    current_directory = os.getcwd()
    model_save_path = os.path.join(current_directory, args.pth_model)
    
    # Load the model
    model_name = args.model_name
    if model_name == "Simple":
        model = SimpleSegmentationModel().to(device)
    elif model_name == "UNet":
        model = SegmentationModel(channels=len(args.channels)).to(device)
    elif model_name == "Pretrained":
        model = pretrained_UNet().to(device)
    
    model.load_state_dict(torch.load(model_save_path))
    print(f"Model {args.pth_model} loaded successfully.")
    
    # Call the evaluation function
    evaluate_model(model, test_dataloader, criterion, device)
