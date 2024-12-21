# README

## Overview

This repository contains the implementation of our project on cell segmentation for the course Deep Learning (02456) at DTU. The project integrates preprocessing, model training, evaluation, and explainability through Grad-CAM, offering a complete pipeline for binary segmentation of cells and nanopillars in brightfield microscopy images.

## Dependencies

To install the dependencies of the project, first create a python environment and install the dependecies following the next commands. The activation is based on a unix environment.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Repository Structure
- **`results.ipynb`**  
  Contains code to reproduce the numerical results presented in the report, including metrics such as Dice Score, IoU, Precision, and Recall for various models and preprocessing strategies.

- **`saliency_maps.ipynb`**  
  This notebook explains the generation of saliency maps using Grad-CAM, including the justification for the interpretability-driven channel selection. The results displayed in Figure 5 of the report can be reproduced using this notebook.

- **`display.ipynb`**  
  Provides visualization of the ground truth and predicted segmentation masks, reproducing the results shown in Figure 3 of the report.

- **`models.py`**  
  Defines the different models used in the project, including the U-Net and Pretrained U-Net architectures.

- **`train.py`**  
  Contains the training function for training segmentation models.

- **`evaluate.py`**  
  Contains the evaluation function for testing the trained models on the test dataset.

- **Preprocessing Files**  
  - **`data_augmentation.ipynb`**: Implements data augmentation techniques for improving model generalization.  
  - **`dataloader.py`**: Contains the dataset loader implementation for managing the input images and masks.  
  - **`evaluate_fft.py`**: Evaluates the FFT baseline and applies thresholding for binary segmentation.  
  - **`fft.ipynb`**: Notebooks for implementing FFT preprocessing and visualization.  

- **HPC Scripts**  
  - **`run_train.sh`**: Script to train a model on the HPC environment.  
  - **`run_evaluate.sh`**: Script to load and evaluate a trained model on the HPC environment.  

- **Subdirectories**
  - **`models/`**  
    Directory containing the `.pth` files of the final trained models used in the report.

  - **`other/`**  
    Contains files related to cross-validation and also to the fft. Note that this directory includes experiments that were not used in the final results.
  

## Notes
- The results, including channel selection and saliency maps, rely on interpretability-driven Grad-CAM insights, which are explained in `saliency_maps.ipynb`.
- All scripts and notebooks are designed to be modular and can be run independently for specific tasks like training, evaluation, preprocessing, and visualization.
- Ensure that the necessary dependencies are installed before running any of the scripts or notebooks. The `requirements.txt` file in the repository contains all required dependencies and can be used to set up the environment.

## Contact
For any questions or issues, please reach out the authors.
