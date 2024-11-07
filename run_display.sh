#!/bin/bash
#BSUB -J display_images          # Job name
#BSUB -n 1                       # Number of cores

# Remove old output files if they exist
# rm -f display_images.out display_images.err

# Run the Python script
python3 display.py --sample-number 8
