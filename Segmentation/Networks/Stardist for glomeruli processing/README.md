A 2D Stardist network was trained on annotations segmenting glomeruli in 2D patches. This training was done using the notebook from ZeroCostDL4Mic (https://github.com/HenriquesLab/ZeroCostDL4Mic/blob/master/Colab_notebooks/StarDist_2D_ZeroCostDL4Mic.ipynb). The trained network's parameters are inside the zipped folder "glomModelIllumCorrected5.zip". Unzipping it allows to use the notebooks in this folder to apply the network and process stacks of LSFM kidney images to segment glomeruli. The requirements.txt file contains packages that need to be installed in your python environment in order to run the code.

There are a series of notebooks in this folder:

1 - Allows to load these weights and apply the network to produce glomeruli segmentations
2 - Binarizes the previously obtained masks, applies a median z-filter to correct skipped frames in the glomeruli segmentation and applies connected components analysis and gives a tif stack and a txt file with glomeruli data as output
3 - Combines the resulting images with masks of the medulla (inner region of the kidney) to mark glomeruli centroids as being inside or outside it