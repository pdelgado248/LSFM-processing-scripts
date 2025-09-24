# LSFM-processing

Scripts for processing Healthy and pathological kidney Light Sheet Fluorescence Microscopy (LSFM) images. Pathological kidneys were affected by Autosomal Dominant Polycistic Kidney Disease (ADPKD). 

Folders in this repository are numbered in order of execution, containing each a different README.md file to explain its contents and allow users to navigate through them. Numbered Jupyter notebooks allow to run the code for each of the sections.

There are two initial folders in the repository’s main page:

>1-Segmentation

It contains codes to segment several regions of the kidney, including:

- Full kidney
- Inner region and cortex
- Inner cavities
- Glomeruli
- Cysts

and obtain several measurements from each of them, producing in the end statistical results and plots.
Inside this folder you can find two subfolders. "Data processing", that contains pathways for segmentation of full kidney, inner regions, cortex and inner cavities, together with analyses of all segmentations. The "Networks" folder, on the other hand, features scripts to apply deep learning methods to segment glomeruli and cysts.

>2-Texture analysis

This folder focuses on texture analysis of 3D patches, cropped from the full LSFM images. Several texture features are extracted from each of these patches, and are then used to train a classifier of pathological and healthy samples. After this, an analysis of importance of each of the features considered is conducted.
Each subfolder in this repository includes as well the corresponding "requirements.txt" file to specify the necessary packages needed to be installed to run the code.”
