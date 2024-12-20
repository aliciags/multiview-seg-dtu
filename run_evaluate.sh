#!/bin/bash
#BSUB -q gpuv100
#BSUB -J evaluate_fft           # Job name
#BSUB -n 1                      # Number of cores
#BSUB -W 2:00                   # Wall-clock time (14 hours here)
#BSUB -R "rusage[mem=8GB]"      # Memory requirements (8 GB here)
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o evaluate_fft.out     # Standard output file
#BSUB -e evaluate_fft.err     # Standard error file

# Activate the environment
# source /zhome/44/2/213836/myenv/bin/activate
source /zhome/2b/8/212341/multiview-seg-dtu/.venv/bin/activate


# Run the Python script
#python3 evaluate.py --pth_model "final_models/segmentation_model_UNet_train100%_channels1.pth" --channels 1
#python3 evaluate.py --pth_model "final_models/segmentation_model_UNet_train100%_channels1_6.pth" --channels 1 6
#python3 evaluate.py --pth_model "final_models/segmentation_model_UNet_train100%_channels1_6_3_4_8.pth" --channels 1 6 3 4 8
#python3 evaluate.py --pth_model "final_models/segmentation_model_UNet_train100%_channels1_6_3_4_8_2_0.pth" --channels 1 6 3 4 8 2 0
#python3 evaluate.py --pth_model "final_models/segmentation_model_UNet_train100%_channels6.pth" --channels 6
#python3 evaluate.py --pth_model "final_models/segmentation_model_UNet_train100%_channels5_6_7.pth" --channels 5 6 7
#python3 evaluate.py --pth_model "final_models/segmentation_model_UNet_train100%_channels4_5_6_7_8.pth" --channels 4 5 6 7 8
#python3 evaluate.py --pth_model "final_models/segmentation_model_UNet_train100%_channels3_4_5_6_7_8_9.pth" --channels 3 4 5 6 7 8 9

#python3 evaluate.py --pth_model "final_models/segmentation_model_UNet_train20%_channels0_1_2_3_4_5_6_7_8_9_10.pth" --channels 0 1 2 3 4 5 6 7 8 9 10
#python3 evaluate.py --pth_model "final_models/segmentation_model_UNet_train40%_channels0_1_2_3_4_5_6_7_8_9_10.pth" --channels 0 1 2 3 4 5 6 7 8 9 10
#python3 evaluate.py --pth_model "final_models/segmentation_model_UNet_train60%_channels0_1_2_3_4_5_6_7_8_9_10.pth" --channels 0 1 2 3 4 5 6 7 8 9 10
#python3 evaluate.py --pth_model "final_models/segmentation_model_UNet_train80%_channels0_1_2_3_4_5_6_7_8_9_10.pth" --channels 0 1 2 3 4 5 6 7 8 9 10

# python3 evaluate.py --pth_model "final_models/segmentation_model_Pretrained.pth" --model_name "Pretrained" --channels 0 1 2 3 4 5 6 7 8 9 10
# python3 evaluate.py --pth_model "final_models/segmentation_model_Simple.pth" --model_name "Simple" --channels 0 1 2 3 4 5 6 7 8 9 10
python3 evaluate_fft.py --channels 0 1 2 3 4 5 6 7 8 9 10
