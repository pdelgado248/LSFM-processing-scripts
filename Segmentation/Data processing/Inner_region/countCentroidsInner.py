from glob import glob
import os
import tifffile as tif
import numpy as np
import pandas as pd
import time


def countCentroidsInner(mainPathInnerMask,mainPathCentroids,type = 'glomeruli'):
    '''
    It calculates the number of elements that are in the inner region of the kidney. 
    These elements come in the form of centroid coordinates previously stored in a txt file,
    and must be compared with the inner region of the kidney binary mask. A new txt file
    is created with the total number of elements and the number of elements that are inside the inner region.

    mainPathInnerMask: str
        Path to the main folder where the inner region masks are stored. The mask NEEDS to be binary,
        with 0 for background and 1 for the inner region.
    mainPathCentroids: str
        Path to the main folder where the centroid coordinates are stored in txt files
    type: str  
        Type of elements that are being analyzed. It can be 'glomeruli' or 'cysts'
    '''
    #Get the main folders from the centroids folder
    mainFolders = glob(f'{mainPathCentroids}/*')


    for mainFolder in mainFolders:
        
        folderName = os.path.basename(mainFolder)

        print('Scanning folder: ', folderName)  

        #Take all txt files in the folder
        txtFileNames = glob(f'{mainFolder}/*.txt')

        #Loop over all the inner masks + txt file pairs
        for i,txtFileName in enumerate(txtFileNames):
            
            #Get the name of the txt file without extras
            if type == 'glomeruli':
                #Extract the corresponding inner mask name
                innerMaskName = os.path.basename(txtFileName).split('_LabelledGlom')[0]
                innerMaskName = f'{innerMaskName}_0.5_Lectine' 
                innerMaskPath = f'{mainPathInnerMask}/{folderName}/kidneyMask-{innerMaskName}_pygorpho_strRad_20'

            elif type == 'cysts':
                #Extract the corresponding inner mask name
                innerMaskName = os.path.basename(txtFileName).split('-imDims')[0]
                if folderName == 'MacroSPIM2':
                    innerMaskName = innerMaskName.split('CystsMask-')[1]
                innerMaskPath = f'{mainPathInnerMask}/{folderName}/kidneyMask-{innerMaskName}_pygorpho_strRad_20'
            
            #Skip the ones without a mask
            if not os.path.exists(innerMaskPath):
                print('No mask found for: ', innerMaskPath)
                continue

            print('Reading mask: ', innerMaskPath)
            #Read the mask
            start_time = time.time()
            innerMask = tif.imread(f'{innerMaskPath}/*')
            print('Elapsed time to load the image: ', (time.time() - start_time)/60, ' min')
            mid_time = time.time()

            maskDims = innerMask.shape

            #Load the txt information as a dataframe
            data = pd.read_csv(txtFileName)

            #Load the centroid coordinates in an array
            coords =  data[['centroid-0','centroid-1','centroid-2']].values
            
            coords = coords.astype(int)  

            #Check if the coordinates are within the mask dimensions
            coords = coords[(coords[:,0] < maskDims[0]) & (coords[:,1] < maskDims[1]) & (coords[:,2] < maskDims[2])]

            #Get the number of centroids
            numCentroids = coords.shape[0]

            #Get the number of centroids that are in the inner region

            numCentroidsInside = np.sum(innerMask[coords[:,0],coords[:,1],coords[:,2]])
            print('Elapsed time to check how many centroids in the inner region: ', (time.time() - mid_time)/60, ' min')

            with open(f'{mainPathInnerMask}/{folderName}/{innerMaskName}-{type}InnerRegion.txt', 'w') as f:
                f.write('numCentroids,numCentroidsInside\n')
                f.write(f'{numCentroids},{numCentroidsInside}')
