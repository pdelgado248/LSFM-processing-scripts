import tifffile as tif
import time
import os
from skimage.measure import regionprops_table
import pandas as pd
import scipy.ndimage as ndi

def measureConComps(maskPath,resultReportFolder,onlyArea = False,reScaleProp=1,startingSlice=0,dimension='3D',inertiaTensEigen = False, typeOfMask='cysts',saveSlices = False):
    '''This function reads an already labeled 3D mask and
    stores the volume and centroid of each component in a txt file.
    if onlyArea is True, only the volume of the components will be stored.
    rescaleProp is a proportion to rescale the image. It is used to reduce the size of the image before processing.
    MissingSlices accounts for the initial slices that should not be numbered
    startingSlice is the number of the first slice that should be saved.
    The variable "dimension" can be '2D' or '3D'. If 3D, a path to a 3D mask is expected.
    If 2D, a path to a folder with 2D slices is expected. TypeOfMask can be glomeruli or
    cysts, which will influence how the folder's name is extracted.
    The variable saveSlices is a boolean that determines if the slices should be saved separate.
    If inertiaTensEigen is True, the inertia tensor eigenvalues of the components will be calculated and saved in the txt file.
    These will in turn serve to calculate the main axes of the components.
    '''
    start_time = time.time()

    #Read the mask
    if dimension == '3D':
        labeled_mask = tif.imread(maskPath)

    elif dimension == '2D':
        labeled_mask = tif.imread(f'{maskPath}/*')

    print('Elapsed time to load the image: ', (time.time() - start_time)/60, ' min')
    mid_time = time.time()

    #if type(labeled_mask[0,0,0]) is not 'uint16':
    #    labeled_mask = labeled_mask.astype('uint16')
    #    print('Elapsed time for uint32 to uint16 conversion: ', (time.time() - start_time)/60, ' min')

    #Get the name of the mask
    if (dimension == '3D') and (typeOfMask == 'glomeruli'):
        im_name = os.path.basename(maskPath).split('_LabelledGlom')[0]
        nameForReport = os.path.basename(maskPath).split('.tif')[0]
        resultReportPath = f'{resultReportFolder}/{nameForReport}-imDims-{labeled_mask.shape[0]}-{labeled_mask.shape[1]}-{labeled_mask.shape[2]}.txt'
    elif (dimension == '2D'):
        im_name = os.path.basename(maskPath)
        resultReportPath = f'{resultReportFolder}/{os.path.basename(maskPath)}-imDims-{labeled_mask.shape[0]}-{labeled_mask.shape[1]}-{labeled_mask.shape[2]}.txt'
    
    if reScaleProp != 1:
        #Rescale the image
        labeled_mask = ndi.zoom(labeled_mask, reScaleProp, order=1)
    
    #Get the properties of the components. 
    #If onlyArea is True, only the volume of the components will be calculated.
    if onlyArea:
        resultsTable = regionprops_table(labeled_mask, properties=('label', 'area'), cache=True)
    else:
        #Otherwise, if inertiaTensEigen is True, the inertia tensor eigenvalues will be calculated.
        if inertiaTensEigen:
            resultsTable = regionprops_table(labeled_mask, properties=('label', 'centroid','area','inertia_tensor_eigvals'), cache=True)     
        
        #Else, the centroid and volume of the components will be calculated.
        else:
            resultsTable = regionprops_table(labeled_mask, properties=('label', 'centroid','area'), cache=True)

    resultsTable = pd.DataFrame(resultsTable)  

    #If there has been a rescaling
    if reScaleProp != 1:
        #Rescale the centroids back to their original size
        resultsTable['centroid-0'] = resultsTable['centroid-0']/reScaleProp
        resultsTable['centroid-1'] = resultsTable['centroid-1']/reScaleProp
        resultsTable['centroid-2'] = resultsTable['centroid-2']/reScaleProp

        #Rescale the volumes back to their original size
        resultsTable['area'] = resultsTable['area']/(reScaleProp**3)

        if inertiaTensEigen:
            #Rescale the inertia tensor eigenvalues back to their original size
            resultsTable['inertia_tensor_eigvals-0'] = resultsTable['inertia_tensor_eigvals-0']/(reScaleProp**2)
            resultsTable['inertia_tensor_eigvals-1'] = resultsTable['inertia_tensor_eigvals-1']/(reScaleProp**2)
            resultsTable['inertia_tensor_eigvals-2'] = resultsTable['inertia_tensor_eigvals-2']/(reScaleProp**2)

    resultsTable.to_csv(resultReportPath)

    print('Elapsed time to save all data in a txt: ', (time.time() - mid_time)/60, ' min')
    mid_time = time.time()

    if saveSlices:
        if reScaleProp != 1:
            #Rescale the image back to its original size
            labeled_mask = ndi.zoom(labeled_mask, 1/reScaleProp, order=1)

        #Create a folder to save the slices of the mask
        slicesPath = f'{os.path.dirname(resultReportPath)}/{im_name}'
        if not os.path.exists(slicesPath):
            os.makedirs(slicesPath)
        
        #Save the mask as a stack of 2D slices
        if typeOfMask == 'glomeruli':
            for i in range(labeled_mask.shape[0]):
                slice = labeled_mask[i,...]
                #Create the number of the slice for its name, adding the missing first slices.
                num = '0000'
                num = num[:-len(str(i+startingSlice))] + (str(i+startingSlice))
                # Generate filename for the slice
                sliceFilename = f'{slicesPath}/{im_name}_SRL_Z{num}_C00.tif'
                # Save the slice as an image
                tif.imwrite(sliceFilename, slice)

        print('Elapsed time to save the mask (in uint16) as a stack of 2D slices: ', (time.time() - mid_time)/60, ' min')

    end_time = time.time()
    print('Elapsed time: ', (end_time - start_time)/60, ' min')
    return