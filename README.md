# LSFM-processing
This repository contains scripts for processing healthy and pathological kidney Light Sheet Fluorescence Microscopy (LSFM) images. Pathological kidneys correspond to samples affected by Autosomal Dominant Polycystic Kidney Disease (ADPKD). 

it is organized into sequentially numbered folders. Each folder contains:

- A dedicated README.md file explaining its contents. 
- Numbered Jupyter notebooks to run the code step by step.
- A requirements.txt file specifying the required packages.
Repository structure

# 1-Segmentation
Contains scripts for segmenting multiple kidney regions. 
Subfolders:
- Data processing: Workflows for segmenting the whole kidney, inner regions, cortex, and inner cavities, along with analysis scripts. 
- Networks: Deep learning pipelines for segmenting glomeruli and cysts. 
These segmentations allow the extraction of multiple measurements, ultimately generating statistical results and plots.
# 2-Texture analysis
Contains scripts for texture analysis of 3D patches cropped from full LSFM images. Extracted texture features are used to:

1 - Train a classifier to distinguish between healthy and pathological samples.

2 - Assess the relative importance of individual features.”

