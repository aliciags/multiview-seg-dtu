#!/bin/bash
#BSUB -q gpuv100
#BSUB -J cv                      # Job name
#BSUB -n 4                       # Number of cores
#BSUB -W 14:00                   # Wall-clock time (14 hours here)
#BSUB -R "rusage[mem=8GB]"       # Memory requirements (8 GB here)
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o cv_%J.out               # Standard output file
#BSUB -e cv_%J.err               # Standard error file

# Activate the environment
source /zhome/2b/8/212341/multiview-seg-dtu/.venv/bin/activate

# Run the Python script
CUDA_LAUNCH_BLOCKING=1 python3 cross_validation.py > output.txt
