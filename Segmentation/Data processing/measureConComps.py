import tifffile as tif
import time
import os
from skimage.measure import regionprops_table, marching_cubes
import pandas as pd
import scipy.ndimage as ndi
import numpy as np
from tqdm import tqdm

def measureConComps(maskPath,resultReportFolder,onlyArea = False,surfaces = False, reScaleProp=1,startingSlice=0,dimension='3D',inertiaTensEigen = False, typeOfMask='cysts',saveSlices = False):
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
        if inertiaTensEigen and not surfaces:
            resultsTable = regionprops_table(labeled_mask, properties=('label', 'centroid','area','inertia_tensor_eigvals'), cache=True)     

        #If surfaces is True, surface area and sphericity will be calculated later using the bounding box.
        if inertiaTensEigen and surfaces:
            resultsTable = regionprops_table(labeled_mask, properties=('label', 'centroid','area','inertia_tensor_eigvals','bbox'), cache=True)

        elif surfaces and not inertiaTensEigen:
            resultsTable = regionprops_table(labeled_mask, properties=('label', 'centroid','area','bbox'), cache=True)

        else:
            resultsTable = regionprops_table(labeled_mask, properties=('label', 'centroid','area'), cache=True)

    resultsTable = pd.DataFrame(resultsTable)  

    # Add surface area and sphericity calculations, only if requested
    if surfaces and not onlyArea:
        # Prepare lists to store new results
        surface_areas = []
        sphericities = []

        def compute_surface_area(mask):
            """Computes surface area of a 3D binary mask using marching cubes."""
            try:
                verts, faces, _, _ = marching_cubes(mask, level=0)
                tri_areas = 0.5 * np.linalg.norm(np.cross(
                    verts[faces[:, 1]] - verts[faces[:, 0]],
                    verts[faces[:, 2]] - verts[faces[:, 0]]
                ), axis=1)
                return np.sum(tri_areas)
            except:
                return np.nan

        # Iterate through each region to calculate surface area and sphericity
        for i, row in tqdm(resultsTable.iterrows(), total=len(resultsTable), desc="Obtaining surface area and sphericity"):

            # Extract label and volume of that specific component
            label_id = row['label']
            volume = row['area']

            # Get bounding box and crop with margin
            mini, minj, mink = int(row['bbox-0']), int(row['bbox-1']), int(row['bbox-2'])
            maxi, maxj, maxk = int(row['bbox-3']), int(row['bbox-4']), int(row['bbox-5'])

            margin = 2
            mini = max(mini - margin, 0)
            minj = max(minj - margin, 0)
            mink = max(mink - margin, 0)
            maxi = min(maxi + margin, labeled_mask.shape[0])
            maxj = min(maxj + margin, labeled_mask.shape[1])
            maxk = min(maxk + margin, labeled_mask.shape[2])

            # Crop region and create binary mask
            cropped = labeled_mask[mini:maxi, minj:maxj, mink:maxk]
            local_mask = (cropped == label_id)

            # Compute surface area using marching cubes
            surface_area = compute_surface_area(local_mask)

            # Compute sphericity
            if not np.isnan(surface_area) and surface_area > 0:
                sphericity = (np.pi ** (1 / 3)) * ((6 * volume) ** (2 / 3)) / surface_area
            else:
                sphericity = np.nan

            # Append results to lists
            surface_areas.append(surface_area)
            sphericities.append(sphericity)


        # Add to DataFrame
        resultsTable['surface_area'] = surface_areas
        resultsTable['sphericity'] = sphericities

    #If there has been a rescaling
    if reScaleProp != 1:
        #Rescale the centroids back to their original size
        resultsTable['centroid-0'] = resultsTable['centroid-0']/reScaleProp
        resultsTable['centroid-1'] = resultsTable['centroid-1']/reScaleProp
        resultsTable['centroid-2'] = resultsTable['centroid-2']/reScaleProp

        #Rescale the volumes back to their original size
        resultsTable['area'] = resultsTable['area']/(reScaleProp**3)

        if inertiaTensEigen and not onlyArea:
            #Rescale the inertia tensor eigenvalues back to their original size
            resultsTable['inertia_tensor_eigvals-0'] = resultsTable['inertia_tensor_eigvals-0']/(reScaleProp**2)
            resultsTable['inertia_tensor_eigvals-1'] = resultsTable['inertia_tensor_eigvals-1']/(reScaleProp**2)
            resultsTable['inertia_tensor_eigvals-2'] = resultsTable['inertia_tensor_eigvals-2']/(reScaleProp**2)

        if surfaces and not onlyArea:
            #Rescale the surface areas back to their original size
            resultsTable['surface_area'] = resultsTable['surface_area']/(reScaleProp**2)
    
    # Remove bbox columns if they exist (to simplify the results table)
    bbox_cols = [col for col in resultsTable.columns if col.startswith('bbox-')]
    resultsTable.drop(columns=bbox_cols, inplace=True)

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