import re
import os
import torch
import random
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.transforms.v2 as transforms

class Well1ImageMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None, channel_indices=None):
        """
        Args:
            image_dir (str): Path to the directory containing images.
            mask_dir (str): Path to the directory containing masks.
            transform (callable, optional): Optional transform to be applied on an image.
            mask_transform (callable, optional): Optional transform to be applied on a mask.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.channels = channel_indices if channel_indices is not None else list(range(11))
        self.sample_dict = self._get_sample_dict()

    def _get_sample_dict(self):
        """
        Organize images by sample ID into a dictionary where the keys are
        sample IDs and values are lists of image file paths corresponding to each sample.
        Each key (sample ID) maps to a list of 11 image paths and 1 mask path.
        """
        sample_dict = {}
        
        # Loop through images in the directory and group by sample ID
        for filename in os.listdir(self.image_dir):
            if filename.endswith("c2_ORG.tif"):  # Filter brightfield images only
                parts = filename.split('_')
                sample_id = parts[4][:3]  # Extract 's08' as the sample ID
                
                # Extract the z-index using regex
                match = re.search(r'z(\d{2})', filename)
                if match:
                    z_index = match.group(1)  # Get the two-digit z-index as a string (e.g., '01')
                    z_idx = int(z_index) - 1  # Convert to integer 0-10 for list indexing

                    if sample_id not in sample_dict:
                        sample_dict[sample_id] = {'images': [None] * 11, 'mask': None}

                    sample_dict[sample_id]['images'][z_idx] = os.path.join(self.image_dir, filename)
                    #print(sample_dict[sample_id]['images'][z_idx])
                    
                    # Determine the corresponding mask path based on sample_id and z06, c1 naming convention
                    if z_index == "06":  # Only one mask for focal point z06
                        mask_filename = f"{'_'.join(parts[:3])}_50locations_{parts[4][:6]}c1_ORG_mask.tiff"
                        #print(mask_filename)
                        mask_path = os.path.join(self.mask_dir, mask_filename)
                        if os.path.exists(mask_path):
                            sample_dict[sample_id]['mask'] = mask_path
                else:
                    print(f"Warning: Could not find z-index in filename {filename}")
        
        # Filter out samples with missing images or mask
        sample_dict = {k: v for k, v in sample_dict.items() if all(v['images']) and v['mask']}
        return sample_dict

    def __len__(self):
        return len(self.sample_dict)

    def __getitem__(self, idx):
        sample_id = list(self.sample_dict.keys())[idx]
        image_paths = self.sample_dict[sample_id]['images']
        mask_path = self.sample_dict[sample_id]['mask']

        # Load the specified images and apply transformations
        images = []
        for i in self.channels:
            img_path = image_paths[i]
            if img_path is not None:
                img = Image.open(img_path).convert("L")
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            else:
                raise FileNotFoundError(f"Image for channel {i} in sample {sample_id} is missing.")

        # Stack images along the channel dimension (C, H, W)
        images = torch.stack(images, dim=0)
        images = images.squeeze(1)

        # Load the mask and apply mask-specific transformations
        if mask_path is None:
            raise FileNotFoundError(f"Mask for sample {sample_id} is missing.")
        mask = Image.open(mask_path).convert("L")
        if self.mask_transform:
            mask = self.mask_transform(mask)
        binary_mask = (mask > 0.5).float()

        return images, binary_mask


class WellsImageMaskDataset(Dataset):
    def __init__(self, image_dirs, mask_dir, transform=None, mask_transform=None, channel_indices=None):
        """
        Args:
            image_dir (str): Path to the directory containing images.
            mask_dir (str): Path to the directory containing masks.
            transform (callable, optional): Optional transform to be applied on an image.
            mask_transform (callable, optional): Optional transform to be applied on a mask.
        """
        self.image_dirs = image_dirs
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.channels = channel_indices if channel_indices is not None else list(range(11))
        self.sample_dict = self._get_sample_dict()

    def _get_sample_dict(self):
        """
        Organize images by sample ID into a dictionary where the keys are
        sample IDs and values are lists of image file paths corresponding to each sample.
        Each key (sample ID) maps to a list of 11 image paths and 1 mask path.
        """
        sample_dict = {}
        well_id = 1

        # Loop through each well directory and process images
        for image_dir in self.image_dirs:
            well_id = well_id + 1
            # Loop through images in the directory and group by sample ID
            for filename in os.listdir(image_dir):
                if filename.endswith("c2_ORG.tif"):  # Filter brightfield images only
                    parts = filename.split('_')
                    sample_id = parts[4][:4]  # Extract 's008' as the sample ID
                    unique_id = f"{well_id}-{sample_id}"
                    #print(unique_id)
    
                    # Extract the z-index using regex
                    match = re.search(r'z(\d{2})', filename)
                    if match:
                        z_index = match.group(1)  # Get the two-digit z-index as a string (e.g., '01')
                        z_idx = int(z_index) - 1  # Convert to integer 0-10 for list indexing
    
                        if unique_id not in sample_dict:
                            sample_dict[unique_id] = {'images': [None] * 11, 'mask': None}
    
                        sample_dict[unique_id]['images'][z_idx] = os.path.join(image_dir, filename)
                        #print(sample_dict[unique_id]['images'][z_idx])
                        
                        # Determine the corresponding mask path based on sample_id and z06, c1 naming convention
                        if z_index == "06":  # Only one mask for focal point z06
                            mask_filename = f"{'_'.join(parts[:4])}_{parts[4][:7]}c1_ORG_mask.tiff"
                            #print(mask_filename)
                            mask_path = os.path.join(self.mask_dir, mask_filename)
                            if os.path.exists(mask_path):
                                sample_dict[unique_id]['mask'] = mask_path
                    else:
                        print(f"Warning: Could not find z-index in filename {filename}")
        
        # Filter out samples with missing images or mask
        sample_dict = {k: v for k, v in sample_dict.items()}
        #print("Well :", sample_dict)
        return sample_dict

    def __len__(self):
        return len(self.sample_dict)

    def __getitem__(self, idx):
        sample_id = list(self.sample_dict.keys())[idx]
        image_paths = self.sample_dict[sample_id]['images']
        mask_path = self.sample_dict[sample_id]['mask']

        # Load the specified images and apply transformations
        images = []
        for i in self.channels:
            img_path = image_paths[i]
            if img_path is not None:
                img = Image.open(img_path).convert("L")
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            else:
                raise FileNotFoundError(f"Image for channel {i} in sample {sample_id} is missing.")

        # Stack images along the channel dimension (C, H, W)
        images = torch.stack(images, dim=0)
        images = images.squeeze(1)

        # Load the mask and apply mask-specific transformations
        if mask_path is None:
            raise FileNotFoundError(f"Mask for sample {sample_id} is missing.")
        mask = Image.open(mask_path).convert("L")
        if self.mask_transform:
            mask = self.mask_transform(mask)
        binary_mask = (mask > 0.5).float()

        return images, binary_mask
        

def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
    dir_path (str or pathlib.Path): target directory
    
    Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def plot_transformed_images(image_paths, transform, n=3, seed=42):
    """Plots a series of random images from image_paths.

    Will open n image paths from image_paths, transform them
    with transform and plot them side by side.

    Args:
        image_paths (list): List of target image paths. 
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 3.
        seed (int, optional): Random seed for the random generator. Defaults to 42.
    """
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f) 
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib 
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0) 
            ax[1].imshow(transformed_image) 
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)

def display_image_stack_with_mask(images, mask, channel_indices):
    """
    Display each selected channel in the image stack as separate images and the corresponding mask in a grid.

    Args:
        images (Tensor): A tensor of shape (C, H, W) representing the multi-channel image stack.
        mask (Tensor): A tensor of shape (1, H, W) representing the binary mask.
        channel_indices (list of int): The original indices of the channels being displayed (for labeling).
    """
    channel_indices = channel_indices if channel_indices is not None else list(range(11))
    num_channels = images.shape[0]
    total_images = num_channels + 1  # Number of images + mask

    # Calculate grid size: rows and columns for the display
    cols = 3
    rows = (total_images + cols - 1) // cols  # Compute rows to fit all images + mask

    # Create a grid for the channels and mask
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    fig.suptitle(f"{num_channels}-Channel Image Stack and Mask", fontsize=16)

    # Flatten axes for easier indexing (handles both 2D and 1D cases)
    axes = axes.flatten()

    # Display each channel image with the correct z-index label
    for i in range(num_channels):
        img_np = images[i].squeeze().numpy()  # Squeeze to shape (H, W)
        axes[i].imshow(img_np, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"Focal Point z{channel_indices[i] + 1:02}")  # Display correct z-index

    # Display the mask in the next available subplot
    mask_np = mask.squeeze().numpy()
    axes[num_channels].imshow(mask_np, cmap='gray')
    axes[num_channels].axis('off')
    axes[num_channels].set_title("Mask")

    # Hide any remaining unused subplots
    for j in range(total_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit the title
    plt.show()

def data_augmentation(dataset):
    # augments data based on dataset
    transform1 = transforms.Compose([
                                    transforms.RandomHorizontalFlip(p=0.75),
                                    transforms.RandomVerticalFlip(p=0.75),
                                    transforms.RandomRotation(degrees=(-180, 180)),                                            
                                    ])
    transform2 = transforms.Compose([
                                    transforms.RandomHorizontalFlip(p=0.75),
                                    transforms.RandomVerticalFlip(p=0.75),
                                    transforms.RandomPerspective(distortion_scale= 0.25, p = 0.8)
                                    ])

    augmented_dataset = []
    for i in range(len(dataset)):
        img_mask = dataset[i]
        img_mask_1 = transform1(img_mask)
        img_mask_2 = transform2(img_mask)
        augmented_dataset.append(img_mask)
        augmented_dataset.append(img_mask_1)
        augmented_dataset.append(img_mask_2)

    return augmented_dataset

def gaussian_filter(img, sigma = 20):
    # Convert to NumPy 
    # try:
    #     img_np = img.numpy()
    # except:
    #     img_np = np.array(img)

    # Fourier Transform
    f_transform = np.fft.fft2(img)  # Compute the FFT
    f_shifted = np.fft.fftshift(f_transform)  # Shift zero frequency to the center

    # Log-transform for better visualization
    # f_magnitude = np.log1p(np.abs(f_shifted))

    # Get center coordinates
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    # Create a Gaussian mask for low-pass filtering  sigma param default 30  
    # Generate a 2D Gaussian kernel
    x = np.linspace(-ccol, ccol, cols)
    y = np.linspace(-crow, crow, rows)
    X, Y = np.meshgrid(x, y)
    gaussian_mask = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    
    # Apply the filter to the frequency domain
    filtered_f_shifted = f_shifted * gaussian_mask
    
    # Visualize the filtered frequency domain
    # filtered_magnitude = np.log1p(np.abs(filtered_f_shifted))

    # Perform the inverse shift
    inverse_shift = np.fft.ifftshift(filtered_f_shifted)
    
    # Inverse Fourier Transform
    filtered_img = np.fft.ifft2(inverse_shift)
    
    # Get the real part and normalize for visualization
    filtered_img = np.abs(filtered_img)
    if np.max(filtered_img) > 0:  # Avoid division by zero
        filtered_img = (filtered_img / np.max(filtered_img) * 255).astype(np.uint8)
    
    return filtered_img


def fft_filter(dataset):
    """
    Apply fft_filter to each tensor in a dataset.

    Parameters:
        dataset (iterable): An iterable collection of data.
        sigma (int): Standard deviation for the Gaussian mask in the filter.

    Returns:
        list: A list of filtered images as NumPy arrays.
    """
    filtered_images = []
    for i in range(len(dataset)):
        img, mask = dataset[i]
        # Loop through each channel
        filtered_channels = []
        for c in range(img.shape[0]):  # img.shape[0] is the number of channels
            channel = img[c]  # Select the c-th channel

            # Convert to NumPy (if not already)
            channel_np = channel.cpu().numpy()
        
            filtered_channel = gaussian_filter(channel_np)
            filtered_channels.append(torch.tensor(filtered_channel, dtype=torch.float32))
        
        filtered_img = torch.stack(filtered_channels, dim=0)
        filtered_images.append((filtered_img, mask))
    
    return filtered_images


def get_dataloader(image_dirs, mask_dir, data_transform, mask_transform, display_sample=False, train_percentage=1.0, channel_indices=None, augmentation=False, fft=False):   
    # plot_transformed_images(image_path_list, transform=data_transform, n=3)
    
    # Initialize dataset and dataloaderz
    train_dataset = WellsImageMaskDataset(image_dirs=image_dirs[1:], mask_dir=mask_dir, transform=data_transform, mask_transform=mask_transform, channel_indices=channel_indices)
    test_dataset = Well1ImageMaskDataset(image_dir=image_dirs[0], mask_dir=mask_dir, transform=data_transform, mask_transform=mask_transform, channel_indices=channel_indices)

    if augmentation:
        train_dataset = data_augmentation(train_dataset)
    elif fft:
        train_dataset = fft_filter(train_dataset)
    elif augmentation and fft:
        train_dataset = data_augmentation(train_dataset)
        train_dataset = fft_filter(train_dataset)

    val_split = 0.2
    # Split the dataset into training and validation based on val_split
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    full_train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    # Further sample the training and validation subsets based on train_percentage
    sampled_train_size = int(train_percentage * len(full_train_subset))
    sampled_train_indices = np.random.choice(full_train_subset.indices, sampled_train_size, replace=False)
    train_subset = Subset(train_dataset, sampled_train_indices)
    
    print("Number of images in the trainset:", len(train_subset))
    print("Number of images in the valset:", len(val_subset))
    print("Number of images in the testset:", len(test_dataset))

    # Create DataLoaders for train and val
    train_dataloader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_subset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)  

    '''
    # Display one sample tensor from trainset for verification
    idx = random.randint(0, len(train_dataset) - 1)
    train_images, train_mask = train_dataset[idx]
    print("Image stack shape:", train_images.shape)  # Should be (11, H, W)
    print("Mask shape:", train_mask.shape)           # Should be (1, H, W)
    # images

    # Display one sample tensor from testset for verification
    idx = random.randint(0, len(test_dataset) - 1)
    test_images, test_mask = test_dataset[idx]
    print("Image stack shape:", test_images.shape)  # Should be (11, H, W)
    print("Mask shape:", test_mask.shape)           # Should be (1, H, W)
    # images

    '''
    if display_sample:
        # Display a random sample from the dataset
        print("First image from trainset:")
        display_image_stack_with_mask(train_dataset[0][0], train_dataset[0][1], channel_indices)
        print("First image from testset:")
        display_image_stack_with_mask(test_dataset[0][0], test_dataset[0][1], channel_indices)
    
    return train_dataloader, val_dataloader, test_dataloader
