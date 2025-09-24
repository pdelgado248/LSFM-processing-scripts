import tifffile as tif
import numpy as np
import os
import time

def measureVolumeBinary(binaryMaskPath):
    """
    Measure the volume of a binary image, stored as 2D
    slices forming a 3D mask and saves it in a txt file.

    binaryMaskPath: str
        The path to the binary image.

    """
    start_time = time.time()
    mask = tif.imread(f'{binaryMaskPath}/*')
    print('Elapsed time to load the image: ', (time.time() - start_time)/60, ' min')
    mid_time = time.time()

    #Get the name of the mask
    im_name = os.path.basename(binaryMaskPath)

    #And the folder where it is
    folder = os.path.dirname(binaryMaskPath)

    #Get the sum of nonzero values
    volume = np.count_nonzero(mask)

    print('Elapsed time calculate nonzero volume: ', (time.time() - mid_time)/60, ' min')
    mid_time = time.time()

    #Save the volume in a txt file, containing the word 'Volume', 
    #a comma, and the volume in voxels of the nonzero values.
    with open(f'{folder}/{im_name}-volume.txt', 'w') as f:
        f.write(f'Volume,{volume}')

    print('Elapsed total time: ', (time.time() - start_time)/60, ' min')
