import raster_geometry as rg
import pygorpho as pg
import tifffile as tif
import time
import os
from glob import glob
import numpy as np
from tqdm import tqdm
import scipy.ndimage as ndi

def morphoProcessingGPU(maskPath, resultPath, strRad, operation, imDimension = '2D', processDimension = '3D', reScaleProp = 1,startingSlice=0):
    '''
    Function to apply 3D morphological operations to a 3D binary mask using GPU. The structuring
    element is a sphere.
    
    Parameters:
    maskPath: str
        Path to the input binary mask.
    resultPath: str
        Path to the output binary mask.
    strRad: int
        Radius in voxels of the structuring element (a sphere).
    operation: str
        Morphological operation to be applied. It can be 'dilate', 'erode', 'open' or 'close'.
    imDimension: str
        Dimension of the mask. It can be '2D' or '3D'. If '2D', the mask is a stack of 2D slices and the
        path to the folder containing the slides is given. If '3D', the mask is a 3D volume and the path
        to the single file is given.
    processDimension: str
        Dimension of the processing. It can be '2D' or '3D'. If '2D', the morphological operation is applied
        to each slice of the mask with a disk. If '3D', the morphological operation is applied to the whole volume
        using a sphere.
    rescaleProp: float 
        Proportion to rescale the image. It is used to reduce the size of the image before processing. It only
        works for 2D images.
    startingSlice: int
        For 3D images, slice to start at for naming the slices.
    '''

    start_time = time.time()

    #Get the name of the mask
    imName = os.path.basename(maskPath)

    #Create a folder to save the resulting slices
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)

    #Read the mask
    if imDimension == '3D':
        mask = tif.imread(maskPath)

    elif imDimension == '2D':
        #Get the names of the images
        imNames = glob(f'{maskPath}/*')
        #Load them as a 3D array
        mask = tif.imread(f'{maskPath}/*')

    print(f'Elapsed time to load the image: {(time.time() - start_time)/60} min')
    mid_time = time.time()     

    #If 3D procesing is selected
    if processDimension == '3D':

        #Rescale the structuring element radius
        strRad = int(strRad*reScaleProp)

        # Create the structuring element (sphere)
        strucEl = 1*(rg.sphere(2*(strRad)+1, strRad))

        #Rescale the mask to make it smaller         
        mask = ndi.zoom(mask, reScaleProp, order=1)

        # Apply the morphological operation
        if operation == 'dilate':
            #Pad the image to avoid border artifacts
            mask = np.pad(mask, strRad+1, mode='constant', constant_values=0)
            mask = pg.flat.dilate(mask, strucEl)
            #Remove the padding
            mask = mask[strRad+1:-strRad-1,strRad+1:-strRad-1,strRad+1:-strRad-1]

        elif operation == 'erode':
            mask = pg.flat.erode(mask, strucEl)
        elif operation == 'open':
            mask = pg.flat.open(mask, strucEl)
        elif operation == 'close':
            #Pad the image to avoid border artifacts
            mask = np.pad(mask, strRad+1, mode='constant', constant_values=0)
            mask = pg.flat.close(mask, strucEl)
            #Remove the padding
            mask = mask[strRad+1:-strRad-1,strRad+1:-strRad-1,strRad+1:-strRad-1]

        #Return to the original size
        mask = ndi.zoom(mask, 1/reScaleProp, order=1)
        print(f'Elapsed time to {operation} the image: {(time.time() - mid_time)/60} min')
        mid_time = time.time()   

        #Save the mask 
        #Loop through each slice of the mask in the first dimension
        if imDimension == '2D' and processDimension == '3D' :
            for i in range(mask.shape[0]):
                slice = mask[i,...]
                # Generate filename for the slice
                sliceFilename = f'{resultPath}/{os.path.basename(imNames[i])}'
                # Save the slice as an image
                tif.imwrite(sliceFilename, slice)
            print(f'Elapsed time to save the image as a stack of 2D slices: {(time.time() - mid_time)/60} min')

        #If dimension is 3D, the file names are created differently
        elif imDimension == '3D' and processDimension == '3D':
            #Remove the extension from the image name
            imName = imName.split('.tif')[0]
            #Initialize number for the slice names
            initNum = '0000'
            for i in range(mask.shape[0]):
                slice = mask[i,...]
                #Create the number of the slice for its name, adding the missing first slices.
                num = initNum[:-len(str(i+startingSlice))] + (str(i+startingSlice))
                # Generate filename for the slice
                sliceFilename = f'{resultPath}/{imName}_{num}.tif'
                # Save the slice as an image
                tif.imwrite(sliceFilename, slice)

            print(f'Elapsed time to save the image as a stack of 2D slices: {(time.time() - mid_time)/60} min')

  

    #If instead the processing is 2D, the morphological operation is applied to each slice
    elif processDimension == '2D': 

        #Rescale the structuring element radius
        strRad = int(strRad*reScaleProp)

        # Create the structuring element (disk)
        strucEl = 1*(rg.circle(2*(strRad)+1, strRad))
        
        #For each slice of the mask
        for i in tqdm(range(mask.shape[0])): 
            
            slice = mask[i,...]

            #Rescale the slice           
            slice = ndi.zoom(slice, reScaleProp, order=1)

            #Apply the morphological operation
            if operation == 'dilate':
                #Pad the image to avoid border artifacts
                mask = np.pad(mask, strRad+1, mode='constant', constant_values=0)
                slice = pg.flat.dilate(slice, strucEl)
                #Remove the padding
                slice = slice[strRad+1:-strRad-1,strRad+1:-strRad-1]
            elif operation == 'erode':
                slice = pg.flat.erode(slice, strucEl)
            elif operation == 'open':
                slice = pg.flat.open(slice, strucEl)
            elif operation == 'close':
                #Pad the image to avoid border artifacts
                mask = np.pad(mask, strRad+1, mode='constant', constant_values=0)
                slice = pg.flat.close(slice, strucEl)
                #Remove the padding
                slice = slice[strRad+1:-strRad-1,strRad+1:-strRad-1]

            #Rescale the slice back to the original size
            slice = ndi.zoom(slice, 1/reScaleProp, order=1)
            
            #Save the slice
            sliceFilename = f'{resultPath}/{os.path.basename(imNames[i])}'
            tif.imwrite(sliceFilename, slice)


    



    
    
    end_time = time.time()
    print(f'Elapsed total time: {(end_time - start_time)/60} min')