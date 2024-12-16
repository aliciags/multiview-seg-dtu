import os
import argparse
from PIL import Image
import matplotlib.pyplot as plt

#image_dir = '/zhome/44/2/213836/02456/project/data/brightfield/Alexa488_Fibroblasts_well2_200locations'
image_dir = '/zhome/70/5/14854/nobackup/deeplearningf24/forcebiology/data/brightfield/Alexa488_Fibroblasts_well2_200locations'
mask_dir = '/zhome/70/5/14854/nobackup/deeplearningf24/forcebiology/data/masks'

parser = argparse.ArgumentParser(description="Display a grid of images and corresponding mask for a specified sample in the well.")
parser.add_argument("--sample-number", type=int, required=True, help="Sample number (1-200) indicating the site in the well")

args = parser.parse_args()

# Validate the sample number (1-200)
if not (1 <= args.sample_number <= 200):
    raise ValueError("Sample number must be between 1 and 200.")

# Format the sample number as a three-digit string (e.g., 8 becomes '008')
sample_str = f"s{str(args.sample_number).zfill(3)}"

image_filenames = [f"Alexa488_Fibroblasts_well2_200locations_{sample_str}z{str(i).zfill(2)}c2_ORG.tif" for i in range(1, 12)]
mask_filename = f"Alexa488_Fibroblasts_well2_200locations_{sample_str}z06c1_ORG_mask.tiff"

# Filter for files in the directory that match the image filenames
image_files = [f for f in os.listdir(image_dir) if f in image_filenames]

if len(image_files) == 0:
    print(f"No matching images found for sample {sample_str} in the specified directory.")
elif not os.path.exists(os.path.join(mask_dir, mask_filename)):
    print(f"No matching mask found for sample {sample_str} in the specified mask directory.")
else:
    images_per_row = 4
    n_images = len(image_files) + 1  # Include the mask as one of the images
    n_rows = max(1, (n_images + images_per_row - 1) // images_per_row)
    image_files = sorted(image_files) # Sort the images to maintain the order from z01 to z11
    fig, axes = plt.subplots(n_rows, images_per_row, figsize=(15, n_rows * 3))
    axes = axes.flatten()

    for idx, image_file in enumerate(image_files):
        img_path = os.path.join(image_dir, image_file)
        img = Image.open(img_path)
        
        axes[idx].imshow(img, cmap="gray")
        axes[idx].axis('off')
        axes[idx].set_title(f"Image {idx + 1}")

    mask_path = os.path.join(mask_dir, mask_filename)
    mask_img = Image.open(mask_path)
    axes[len(image_files)].imshow(mask_img, cmap="gray")
    axes[len(image_files)].axis('off')
    axes[len(image_files)].set_title("Mask")

    # Hide any remaining empty subplots
    for j in range(len(image_files) + 1, len(axes)):
        axes[j].axis('off')

    # Adjust layout and save the figure to a file
    plt.tight_layout()
    output_path = f"public/output_image_grid_{sample_str}.png"
    plt.savefig(output_path, dpi=300)  # Save as a high-resolution image

    print(f"Image grid saved to {output_path}")
