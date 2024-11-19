import os
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
import matplotlib.pyplot as plt
import re
from PIL import Image

class Well1ImageMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
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
        # Retrieve sample ID and paths
        sample_id = list(self.sample_dict.keys())[idx]
        image_paths = self.sample_dict[sample_id]['images']
        mask_path = self.sample_dict[sample_id]['mask']
        
        # Load the 11 images and apply transformations
        images = []
        for img_path in image_paths:
            img = Image.open(img_path).convert("L")  # Convert to grayscale if needed
            if self.transform:
                img = self.transform(img)
            images.append(img)
        
        # Stack images along the channel dimension to create (11, H, W)
        images = torch.stack(images, dim=0)
        images = images.squeeze(1)

        # Load the mask and apply mask-specific transformations
        mask = Image.open(mask_path).convert("L")
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return images, mask  # Return both the 11-channel image stack and the mask


class WellsImageMaskDataset(Dataset):
    def __init__(self, image_dirs, mask_dir, transform=None, mask_transform=None):
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
        # Retrieve sample ID and paths
        sample_id = list(self.sample_dict.keys())[idx]
        image_paths = self.sample_dict[sample_id]['images']
        mask_path = self.sample_dict[sample_id]['mask']
        
        # Load the 11 images and apply transformations
        images = []
        for img_path in image_paths:
            img = Image.open(img_path).convert("L")  # Convert to grayscale if needed
            if self.transform:
                img = self.transform(img)
            images.append(img)
        
        # Stack images along the channel dimension to create (11, H, W)
        images = torch.stack(images, dim=0)
        images = images.squeeze(1)

        # Load the mask and apply mask-specific transformations
        mask = Image.open(mask_path).convert("L")
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return images, mask  # Return both the 11-channel image stack and the mask
        

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

def display_image_stack_with_mask(images, mask):
    """
    Display each channel (focal point) in the 11-channel image stack as separate images
    and the corresponding mask in a 4x3 grid.
    
    Args:
        images (Tensor): A tensor of shape (11, H, W) representing the 11-channel image stack.
        mask (Tensor): A tensor of shape (1, H, W) representing the binary mask.
    """
    num_channels = images.shape[0]  # Should be 11 for this dataset
    
    # Create a 4x3 grid for 11 channels + 1 mask
    fig, axes = plt.subplots(4, 3, figsize=(12, 12))
    fig.suptitle("11-Channel Image Stack and Mask", fontsize=16)

    for i in range(num_channels):
        # Convert each channel to numpy and remove the singleton dimension
        img_np = images[i].squeeze().numpy()  # Squeeze to shape (H, W)
        
        # Display each focal point image
        row, col = divmod(i, 3)
        axes[row, col].imshow(img_np, cmap='gray')
        axes[row, col].axis('off')
        axes[row, col].set_title(f"Focal Point z{i + 1:02}")

    # Display the mask in the last position (row=3, col=2)
    mask_np = mask.squeeze().numpy()  # Squeeze to shape (H, W)
    axes[3, 2].imshow(mask_np, cmap='gray')
    axes[3, 2].axis('off')
    axes[3, 2].set_title("Mask")

    # Hide any unused subplots
    for j in range(num_channels + 1, 12):
        row, col = divmod(j, 3)
        axes[row, col].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit title
    plt.show()

def get_dataloader(image_dirs, mask_dir, data_transform, mask_transform):   
    # plot_transformed_images(image_path_list, transform=data_transform, n=3)
    
    # Initialize dataset and dataloader
    train_dataset = WellsImageMaskDataset(image_dirs=image_dirs[1:], mask_dir=mask_dir, transform=data_transform, mask_transform=mask_transform)
    test_dataset = Well1ImageMaskDataset(image_dir=image_dirs[0], mask_dir=mask_dir, transform=data_transform, mask_transform=mask_transform)
    print("Number of images in the trainset:", len(train_dataset))
    print("Number of images in the testset:", len(test_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
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
        
    # Display a random sample from the dataset
    display_image_stack_with_mask(train_images, train_mask)
    display_image_stack_with_mask(test_images, test_mask)
    '''
    
    return train_dataloader, test_dataloader
