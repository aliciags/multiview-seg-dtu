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
# source /zhome/44/2/213836/myenv/bin/activate
source /zhome/2b/8/212341/multiview-seg-dtu/.venv/bin/activate

# Run the Python script
#python3 train.py --num_epochs 20 --model_name "UNet" --channels 1
#python3 train.py --num_epochs 20 --model_name "UNet" --channels 1 6
#python3 train.py --num_epochs 20 --model_name "UNet" --channels 1 6 3 4 8
#python3 train.py --num_epochs 20 --model_name "UNet" --channels 1 6 3 4 8 2 0
#python3 train.py --num_epochs 20 --model_name "UNet" --train_percentage 0.2
#python3 train.py --num_epochs 20 --model_name "UNet" --train_percentage 0.4
#python3 train.py --num_epochs 20 --model_name "UNet" --train_percentage 0.6
#python3 train.py --num_epochs 20 --model_name "UNet" --train_percentage 0.8
python3 train.py --num_epochs 20 --model_name "UNet" --train_percentage 1.0 --fft True
