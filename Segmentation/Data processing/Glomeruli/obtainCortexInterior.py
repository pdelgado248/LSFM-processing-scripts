import numpy as np
from shapely.geometry import MultiPoint, Polygon
from skimage.draw import polygon
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay
from skimage.measure import label
import pandas as pd
import os
from glob import glob
import tifffile as tif
import alphashape
from tqdm import tqdm
import time
from stl import mesh

def obtainCortexInterior(glomeruliDataPath,origImPath,resultPath,alpha=0,mode = '2D',glomDist = 60):
    '''
    This function reads a txt file containing the coordinates of the centroids of the glomeruli
    and returns a binary mask of the inner area at the center of the kidney (the kidney area that is
    not the cortex) to be later combined with a full kidney mask and obtain the cortex. The 3D mode
    calculates a 3D alpha shape (with parameter alpha) of the glomeruli centroids, while the 2D mode 
    rounds each z-coordinate of the centroids to an integer number, assigning it a specific slice. 
    Then, it goes slice by slice building for each of them a 2D alpha shape (with parameter alpha)
    from these rounded centroids. The 2D alpha shapes are combined to create the 3D result.

    In both cases, once the alpha shape is obtained as 1s, it is inverted to get the inner region + background.
    Connected components are then extracted to get rid of the background (component that is touching the borders).
    The resulting mask is the inner 3D gap.

    - glomeruliDataPath: path to the txt file containing the glomeruli centroids
    - origImPath: path to the original images to get the names to save final slices
    - resultPath: path to save the slices of the resulting mask
    - glomDist: distance in pixels to consider a glomerulus as a neighbor of a pixel
    - alpha: parameter for the alpha shape
    - mode: '3D' or '2D' to calculate the alpha shape in 3D or 2D, respectively
    '''

    start_time = time.time()
    #Extract the image dimensions, that should be specified at the end of the .txt file name as
    #'-imDims-<i>-<j>-<k>.txt'
    imShape = glomeruliDataPath.split('-imDims-')[1].split('.txt')[0].split('-')        
    imShape = np.array([int(i) for i in imShape])

    print(f'The image has {imShape[0]} slices') 
    print('imShape:',imShape)   

    origImList = glob(origImPath+'/*')

    #Create a folder to save the slices of the mask
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)

    #Load the data as a pandas dataframe
    data = pd.read_csv(glomeruliDataPath)
        
    if mode=='3D':
        #Skip the header (first line)
        for i,line in tqdm(data.iterrows()):

            # Create a centroids array
            centroidsArray = np.array([line['centroid-0'], line['centroid-1'], line['centroid-2']])

            #If it is the first line, create a numpy array to store the centroids
            if i==0:
                centroids = centroidsArray
            #Else, add a row to that same array
            else:
                centroids = np.vstack((centroids, centroidsArray))

        '''
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Version 1: compute the 3D alpha shape of all centroids and save it as an stl file~
        #(throws an error)
        
        # Compute the alpha shape
        

        print('centroids.shape:',centroids.shape)
        print('centroids:',centroids) 

        alpha_shape = alphashape.alphashape(centroids, alpha)

        # Extract the vertices and faces from the alpha shape
        def extract_vertices_faces(alpha_shape):
            if alpha_shape.geom_type == 'Polygon':
                vertices = np.array(alpha_shape.exterior.coords)
                faces = Delaunay(vertices).simplices
            elif alpha_shape.geom_type == 'MultiPolygon':
                vertices = np.concatenate([np.array(polygon.exterior.coords) for polygon in alpha_shape.geoms])
                faces = np.concatenate([Delaunay(np.array(polygon.exterior.coords)).simplices for polygon in alpha_shape.geoms])
            return vertices, faces

        vertices, faces = extract_vertices_faces(alpha_shape)

        # Create the mesh
        mesh_data = np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype)
        for i, f in enumerate(faces):
            for j in range(3):
                mesh_data['vectors'][i][j] = vertices[f[j],:]

        # Save the mesh to an STL file
        alpha_shape_mesh = mesh.Mesh(mesh_data)
        alpha_shape_mesh.save(resultPath)
        '''

        '''
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Version 2 - Iterate through all the pixels of the image and calculate the distance to the
        #closest glomerulus. If the distance is less than a threshold, the pixel is considered part of
        #the mask (saturates the RAM)

        # Generate grid indices for each dimension and flatten them
        print('Generating grid indices') 
        i_indices, j_indices, k_indices = np.indices(imShape,dtype='uint16')
        i_indices = i_indices.flatten()
        j_indices = j_indices.flatten()
        k_indices = k_indices.flatten() 
        

        print('Calculating the 3D alpha shape')
        #Initialize an empty mask (all 1s) and flatten it to use only 1 loop
        result_mask = np.ones(imShape,dtype='uint8')
        result_mask = result_mask.flatten()
        
        
        for i in tqdm(range(imShape[0])):
            for j in tqdm(range(imShape[1])):
                for k in range(imShape[2]):
                    coords = np.array([i, j, k],dtype=float)
                    minDist = np.min(np.linalg.norm(centroids - coords, axis=1))
                    if minDist < glomDist:
                        result_mask[i,j,k] = 0

        
        for count in tqdm(range(len(result_mask))):
            coords = np.array([i_indices[count], j_indices[count], k_indices[count]],dtype=float)
            minDist = np.min(np.linalg.norm(centroids - coords, axis=1))
            if minDist < glomDist:
                result_mask[count] = 0
        
        #Reshape the mask to its original shape
        result_mask = result_mask.reshape(imShape)
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Version 3 - Another attempt to calculate the 3D alpha shape (not working)
        # Create a MultiPoint object of the centroids
        points = MultiPoint(centroids)

        # Compute Delaunay triangulation of the points
        tri = Delaunay(points)

        # Create alpha shape by thresholding the Delaunay triangulation
        alpha_shape = unary_union([points[triangle] for triangle in tri.vertices \
                                    if points[triangle].length < alpha])
        
        #Create already inverted binary mask (0s are the alpha shape)
            
        result_mask = np.ones(imShape)
        for pol in polygonize(alpha_shape):
            coords = np.array(pol.exterior.coords.xy).T
            rr, cc, zz = pol(coords[:, 1], coords[:, 0],coords[:, 2])
            result_mask[rr, cc, zz] = 0

        #Connected components of the mask
        result_mask = label(result_mask)
        #Get the background label, the label touching the border
        labelBack = result_mask[0,0,0]

        result_mask[result_mask == labelBack] = 0
        result_mask[result_mask != 0] = 1
        

        #Save the mask as a stack of 2D slices
        for i in range(result_mask.shape[0]):
            slice = result_mask[i,...]
            # Generate filename for the slice
            sliceFilename = origImList[i]
            # Save the slice as an image
            tif.imwrite(sliceFilename, slice)
        '''

    elif mode=='2D':

        print('Loading centroids to an array and rounding the z-coordinate')
        #Store the centroids
        for i,line in tqdm(data.iterrows()):

            # Append to the centroids array, rounding the first coordinate to adjust to the slice 
            centroidsArray = np.array([round(line['centroid-0']), line['centroid-1'], line['centroid-2']])
            
            #If it is the first line, create a numpy array to store the centroids
            if i==0:
                centroids = centroidsArray
            #Else, add a row to that same array
            else:
                centroids = np.vstack((centroids, centroidsArray))
        #print('Centroids:',centroids)
        print('Creating the 2D alpha shapes for each slice')
        #For each slice
        for i in tqdm(range(imShape[0])):
            #Get the centroids of the glomeruli in that slice, excluding the z-coordinate
            centroids2D = centroids[centroids[:, 0]==float(i)][:,1:]
            #print('centroids2D: ',centroids2D)
            if (centroids2D.shape[0] < 3) or np.all(centroids2D[:, 0] == centroids2D[0, 0])\
                or np.all(centroids2D[:, 1] == centroids2D[0, 1]):
                # Create an empty mask if there are less than 3 glomeruli in the slice or if
                # the glomeruli are collinear. In both cases an alpha shape can't be generated
                result_mask2D = np.zeros(imShape[1:])

            else:               
                # Compute the alpha shape
                alpha_shape2D = alphashape.alphashape(centroids2D, alpha)

                # Create already inverted binary mask (0s are the alpha shape)
                result_mask2D = np.ones(imShape[1:], dtype=np.uint8)
                
                if isinstance(alpha_shape2D, Polygon):
                    exterior_coords = np.array(alpha_shape2D.exterior.coords)
                    rr, cc = polygon(exterior_coords[:, 1], exterior_coords[:, 0])
                    rr = np.clip(rr, 0, result_mask2D.shape[0] - 1)
                    cc = np.clip(cc, 0, result_mask2D.shape[1] - 1)
                    result_mask2D[rr, cc] = 0

                elif isinstance(alpha_shape2D, MultiPoint):
                    for geom in alpha_shape2D:
                        exterior_coords = np.array(geom.exterior.coords)
                        rr, cc = polygon(exterior_coords[:, 1], exterior_coords[:, 0])
                        rr = np.clip(rr, 0, result_mask2D.shape[0] - 1)
                        cc = np.clip(cc, 0, result_mask2D.shape[1] - 1)
                        result_mask2D[rr, cc] = 0

                # Connected components of the mask
                result_mask2D = label(result_mask2D, connectivity=1)
                # Get the background label, the label touching the border
                labelBack = result_mask2D[0, 0]
                
                # Remove the background label from the mask
                #result_mask2D[result_mask2D == labelBack] = 0
                result_mask2D[result_mask2D != 0] = 1

            #Save the 2D mask

            sliceFilename = f'{resultPath}/{os.path.basename(origImList[i])}'

            result_mask2D = result_mask2D.astype('uint8')
            tif.imwrite(sliceFilename, result_mask2D)
    
    end_time = time.time()
    print('Elapsed time: ', (end_time - start_time)/60, ' min')

