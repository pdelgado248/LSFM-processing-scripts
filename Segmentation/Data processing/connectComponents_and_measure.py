import numpy as np
import glob
import tifffile as tif
import time
import os

from skimage.measure import label
import matplotlib.pyplot as plt

def connectComponents_and_measure(maskFolder,resultFolder,resultReportFolder,sizeThres):
    
    #Loop through the 2D slices of a 3D binary mask and connect the components
    #in 3D. This script should be used to avoid loading the whole 3D mask in memory.
    
    start_time = time.time()

    #Variable to count the number of slices removed (to update centroid coordinates for saving)
    removedSlicesCount = 0

    #Get the list of files in the folder
    fileList = sorted(glob.glob(maskFolder+'/*'))

    #Create the result folder if it does not exist
    if not os.path.exists(resultFolder):
        os.makedirs(resultFolder)

    resultReportPath = resultReportFolder + '/'+os.path.basename(maskFolder)+'.txt'
    with open(resultReportPath, 'w') as file:
        # Write the header of the txt file where data will be saved
        file.write('label,centroid_i,centroid_j,centroid_k,volume\n')

    # Initialize an array to hold the slices

    #Read and label each connected component from the first slice
    sampleSection_np = label(tif.imread(fileList[0]))
    
    #Connected components below a size threshold in the 2D slice are neglected+
    removedCount = 0
    for label_num in np.unique(sampleSection_np)[1:]:
        if np.sum(sampleSection_np == label_num) < sizeThres:
            sampleSection_np[sampleSection_np == label_num] = 0
            removedCount += 1

    print(f'Slice 0 labeled: {np.max(sampleSection_np)} components')
    print(f'{removedCount} removed components')
    sampleSection_np = sampleSection_np.astype('uint16')

    #Save the first slice
    sliceToSave = 0
    name = os.path.basename(fileList[sliceToSave])
    tif.imsave(f'{resultFolder}/{name}', sampleSection_np)

    sampleSection_np = sampleSection_np[np.newaxis,:,:]

    #print('np.unique(Image_0): ',np.unique(sampleSection_np))
    


    #List to store the labels that have been completed
    completedLabelsList = []    

    for i, filename in enumerate(fileList[1:]):

        # Read the image file
        image = tif.imread(filename)

        #Label the current slice
        image = label(image)
                    
        print(f'Slice {i+1} labeled: {np.max(image)} components')
        #Obtain the previous slice
        prevSlice = sampleSection_np[-1,:,:]

        #Add the maximum value of the previous slice to the current slice to avoid
        #overlapping labels
        image =  image.astype('uint16')
        image[image>0] = image[image>0] + np.max(sampleSection_np)

        #Paint the overlapping connected components with the same label
        overlCount = 0
        #Variable to count how many overlapped regions are found, and correct the remaining labels accordingly
        #to avoid gaps in the labeling 
        subtractIndex = 0
        #To count the number of small components removed
        removedCount = 0
        #Check every label in the current slice to see if it overlaps with any label in the previous slice  
        for label_num in np.unique(image)[1:]: 
            
            #print('label_num: ',label_num)
            #Connected components below a size threshold in the 2D slice are neglected
            if np.sum(image==label_num) < sizeThres:
                image[image == label_num] = 0
                removedCount += 1
                subtractIndex+=1

            else:
                #Get all the values that overlap with the current label
                overlap = prevSlice[image == label_num]
                #Remove 0s
                overlap = overlap[overlap > 0]
                
                if overlap.size > 0:
                    #If something other than background overlaps, paint the current label with the most overlapping label
                    values, counts = np.unique(overlap, return_counts=True)
                    most_frequent_value = values[np.argmax(counts)]
                    overlCount += 1
                    image[image == label_num] = most_frequent_value
                    subtractIndex+=1
                
                else:
                    #Correct the remaining labels to avoid gaps
                    image[image == label_num] = label_num - subtractIndex
        
        #Save the modified mask
        sliceToSave += 1
        name = os.path.basename(fileList[sliceToSave])
        tif.imsave(f'{resultFolder}/{name}', image)
    
        #Check nonzero components of the previous slice to see if any component has finished
        for label_num_prev in np.unique(prevSlice)[1:]:
            #print('label_num_prev: ',label_num_prev)
            #print(f'Checking label {label_num_prev} from previous slice')
            #If the label is not in the current slice, it has finished
            #print('label_num_prev not in np.unique(image)',label_num_prev not in np.unique(image))
            #print('np.unique(image): ',np.unique(image))

            if label_num_prev not in np.unique(image):

                binary_mask = 1*(sampleSection_np == label_num_prev)
                labelToSave = label_num_prev
                coords_i, coords_j, coords_k = np.nonzero(binary_mask)

                #The first coordinate is corrected to account for the slices removed from the moving array. 
                centroidToSave_i = np.mean(coords_i)+removedSlicesCount
                centroidToSave_j = np.mean(coords_j)
                centroidToSave_k = np.mean(coords_k)
                volumeToSave = np.sum(binary_mask)

                #Store the completed label in the list of completed labels
                completedLabelsList.append(labelToSave)

                #print(f'Component {label_num_prev} finished')
                #print(f'Volume: {volumeToSave}, centroid: {centroidToSave_i},{centroidToSave_j},{centroidToSave_k},\
                #        label: {labelToSave}')
                # Open the file to write the results report
                with open(resultReportPath, 'a') as file:    
                    # Assuming this part is inside a loop where you calculate the above variables
                    # Write the data for each label
                    file.write(f'{labelToSave},{centroidToSave_i},{centroidToSave_j},{centroidToSave_k},{volumeToSave}\n')

                #keepSlice will be True if there is atg least one nonzero element in the unique
                #values of the first 2D slice of sampleSection_np that is not already completed.
                #That would mean the slice has to be kept, since it contains a label that has not
                #been saved yet. Lower labels should correspond to labels already saved.
                firstSliceLabels = np.unique(sampleSection_np[0,:,:])[1:]
                toBeClosed = np.isin(firstSliceLabels, completedLabelsList,invert=True)
                
                #print(f'Waiting to be closed: {firstSliceLabels[toBeClosed]}')

                keepSlice = any(toBeClosed)
                
                
                #This loop removes the first slices of the array if the slice can be deleted, continue the loop
                while not keepSlice:
                    print(f'First slice removed, current array depth: {sampleSection_np.shape[0]}')
                    #Save and remove the first slice
                    sampleSection_np = sampleSection_np[1:,:,:]
                    #Recalculate keepSlice
                    keepSlice = np.any(np.unique(sampleSection_np[0,:,:])[1:] > label_num_prev)

                    removedSlicesCount += 1
                
        print(f'{overlCount} overlapped components')   
        print(f'{removedCount} removed components')             
        #tif.imsave(f'{resultFolder}/sampleSection_np_{i+1}',sampleSection_np)
        #print('np.unique(new image): ',np.unique(image))
        # Add the image to the list of slices
        sampleSection_np = np.append(sampleSection_np,[image],axis=0)



    #The result is a slice by slice 3D

    end_time = time.time()
    print('Elapsed time: ', (end_time - start_time)/60, ' min')
