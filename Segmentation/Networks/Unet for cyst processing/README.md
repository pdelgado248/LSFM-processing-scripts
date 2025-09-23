A 2D Unet network was trained on annotations segmenting cysts in 2D full kidney slices. The trained network's parameters are inside the zipped folder "Cyst_PretrainedModel.zip". Unzipping it allows to use the notebooks in this folder to apply the network. The requirements.txt file contains packages that need to be installed in your python environment in order to run the code.

There are two notebooks in this folder:

1 - Loads the Unet network trained for cyst segmentation and applies it to LSFM images, producing cyst masks

2 - Processes the masks by closing and particle removal to improve the results