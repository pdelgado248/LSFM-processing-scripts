
Scripts for segmentation of the full kidney, inner region and cortex, inner cavities, glomeruli and cysts.
The code for pygorpho needs to be downloaded and installed in the main folder (https://github.com/patmjen/pygorpho).

Different folders are focused on processing different regions of the kidney. Scripts outside the folder have general purpose functions, and all of them can be run from the notebooks "runScripts.ipynb", "runScripts 2.ipynb" and "obtainFinalData.ipynb".




Texture analysis folder:

Script to extract texture features from 3D patches of the images and, using them, train a classifier of pathological and healthy patches.