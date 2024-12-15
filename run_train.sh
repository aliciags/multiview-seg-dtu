#!/bin/bash
#BSUB -q gpuv100
#BSUB -J train_seg_model         # Job name
#BSUB -n 4                       # Number of cores
#BSUB -W 14:00                   # Wall-clock time (14 hours here)
#BSUB -R "rusage[mem=8GB]"       # Memory requirements (8 GB here)
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o train_seg_model_%J.out  # Standard output file
#BSUB -e train_seg_model_%J.err  # Standard error file

# Activate the environment
source /zhome/44/2/213836/myenv/bin/activate

# Run the Python script
python3 train.py --num_epochs 20 --model_name "UNet"
python3 train.py --num_epochs 20 --model_name "Simple"
python3 train.py --num_epochs 20 --model_name "Pretrained"
