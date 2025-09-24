import numpy as np
import glob
import tifffile as tif
import time
import os
from skimage.measure import label
from tqdm import tqdm


def connectComponents(maskFolder,resultFolder):
    
    #Loop through the 2D slices of a 3D binary mask and connect the components
    #in 3D. This script should be used to avoid loading the whole 3D mask in memory.
    
    #Create the result folder if it does not exist
    if not os.path.exists(resultFolder):
        os.makedirs(resultFolder)
    
    start_time = time.time()

    #Get the list of files in the folder
    fileList = sorted(glob.glob(maskFolder+'/*'))

    # Initialize an array to hold the slices

    print(f'The image has {len(fileList)} slices')

    #Read and label each connected component from the first slice
    sampleSection_np = label(tif.imread(fileList[0]))

    #print('Slice 0 labeled') 
    sampleSection_np = sampleSection_np.astype('uint32')

    '''
    #Check for too small 2D connected components and remove them. Update the labels
    #to avoid gaps
    subtractIndex = 0
    if sizeThres > 0:   
        for label_num in np.unique(sampleSection_np)[1:]:
            if np.sum(sampleSection_np == label_num) < sizeThres:
                sampleSection_np[sampleSection_np == label_num] = 0
                subtractIndex += 1
            else:
                sampleSection_np[sampleSection_np == label_num] = label_num - subtractIndex
    '''


    #Save the first slice
    sliceToSave = 0
    name = os.path.basename(fileList[sliceToSave])
    tif.imsave(f'{resultFolder}/{name}', sampleSection_np, bigtiff=True)

    #Initialize the maximum label in the previous slices
    maxPrevLabel = np.max(sampleSection_np)

    #Create the array to sweep the image, adding an empty 3rd dimension
    sampleSection_np = sampleSection_np[np.newaxis,...]


    sliceToRemove = 0

    print('Connecting components...')
    for i, filename in tqdm(enumerate(fileList[1:])):
        
        # Read the image file
        image = tif.imread(filename)

        image = label(image)            
        #print(f'Slice {i+1} labeled')

        #If it is not the first slice, paint the overlapping connected components with the same label
        
        overlCount = 0

        #Accounts for the number of neglected components
        subtractIndex = 0

        #Extract the previous slice
        prevSlice = sampleSection_np[-1,...]
        #Update the maximum label in the previous slices
        maxPrevLabel = np.max([np.max(prevSlice),maxPrevLabel])

        #Correct the image labels so that they don't repeat
        image =  image.astype('uint32')
        image[image>0] = image[image>0] + maxPrevLabel
    
        #For each nonzero component in the current slice
        for label_num in np.unique(image)[1:]: 
                     
            #Get all the values that overlap with the current label
            overlap = prevSlice[image == label_num]
            #Remove 0s
            overlap = overlap[overlap > 0]
            
            if overlap.size > 0:
                #If something other than background overlaps, paint the current label with the most overlapping label
                values, counts = np.unique(overlap, return_counts=True)
                most_frequent_value = values[np.argmax(counts)]
                image[image == label_num] = most_frequent_value
                overlCount += 1
                subtractIndex += 1

            else:
                #Correct the remaining labels to avoid gaps
                image[image == label_num] = label_num - subtractIndex
  
        #Save the modified mask
        sliceToSave += 1
        name = os.path.basename(fileList[sliceToSave])
        
        tif.imsave(f'{resultFolder}/{name}', image, bigtiff=True)
        
        #print(f'{overlCount} overlapped components')    


        #print('image.shape: ',image.shape)  
        #print('sampleSection_np.shape: ',sampleSection_np.shape)    
        # Add the image to the list of slices
        sampleSection_np = np.concatenate((sampleSection_np,image[np.newaxis,...]),axis=0)

        if i >= 5:
            #Remove the first slice
            sampleSection_np = sampleSection_np[1:,...]
            sliceToRemove += 1


    #The result is a slice by slice 3D

    end_time = time.time()
    print('Elapsed time: ', (end_time - start_time)/60, ' min')
