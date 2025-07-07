#To be run in the workenvpy3.8 conda environment
import argparse
import numpy as np
import os
import tifffile as tif
import raster_geometry as rg
import pygorpho as pg
import matplotlib.pyplot as plt
import time
from glob import glob
from scipy.ndimage import label
from scipy.ndimage.measurements import sum as ndi_sum
from tqdm import tqdm

class process_fullKidneyMasks:
    def __init__(self, args):

      self.strRad = args.strRad
      self.inputFolder = args.inputFolder
      self.resultFolder = args.resultFolder
      self.kidneyThres = args.kidneyThres
      self.vessThres = args.vessThres
      self.sliceToVis = args.sliceToVis
      self.sliceBySliceThresholds = args.sliceBySliceThresholds
      self.sliceBySliceErodeKidneyMask = args.sliceBySliceErodeKidneyMask
      self.radiusKidneyErosion = args.radiusKidneyErosion
      self.processDimension = args.processDimension

    def kidneyVesselsMasks(self,im,sliceBySliceThresholds,kidneyThres,vessThres):
      """
      Threshold the 3D LSFM image of a full kidney to get both the full kidney (pixels with intensity
      higher than kidneyThres) and the vessels/gaps/background (pixels with intensity lower than vessThres)
      """

      print('Applying kidneyVesselsMask')

      kidneyThres = kidneyThres if kidneyThres >= 0 else np.max(im)/2
      vessThres = vessThres if vessThres >= 0  else np.max(im)/2

      if sliceBySliceThresholds:
            kidneyMask = np.zeros(im.shape,dtype='uint8')
            vesselsMask = np.zeros(im.shape,dtype='uint8')
            for i in range(im.shape[0]):
                  kidneyMask[i, im[i,:,:] >= kidneyThres] = 1
                  vesselsMask[i, im[i,:,:] < vessThres] = 1
      else:
            kidneyMask = 1*(im >= kidneyThres)
            kidneyMask = kidneyMask.astype('uint8')
            vesselsMask = 1*(im < vessThres)
            vesselsMask = vesselsMask.astype('uint8')

      return kidneyMask,vesselsMask
    
    def combineKidneyVesselsMasks(self,sliceBySliceErodeKidneyMask,kidneyMask,radiusKidneyErosion,vesselsMask):
      """
      Erode the full kidney mask and multiply it by the vessels mask to get the contents inside the kidney
      """

      print('Applying combineKidneyVesselsMasks')
      
      if sliceBySliceErodeKidneyMask:
            strucEl = 1*(rg.circle(2*(radiusKidneyErosion)+1, radiusKidneyErosion))
            kidneyMask2 = np.zeros(kidneyMask.shape,dtype='uint8')

            for i in range(kidneyMask.shape[0]):
                  kidneyMask2[i,:,:] = pg.flat.erode(kidneyMask[i,:,:], strucEl)
            
      else: 
            strucEl = 1*(rg.sphere(2*(radiusKidneyErosion)+1, radiusKidneyErosion))
            kidneyMask2 = pg.flat.erode(kidneyMask, strucEl)
            kidneyMask2 = kidneyMask2.astype('uint8')


      #Multiply the kidney mask by the vessels mask to keep only the contents
      #inside the kidney
      
      vesselsMask = vesselsMask*kidneyMask2

      return kidneyMask2, vesselsMask
    

    def padStrRad(self,mask,strRad,mode = 'direct'):
      """
      Pad a 3D matrix adding 0s at the end on
      all directions based on a structural element's
      radius, strRad
      """

      if mode == 'direct':
            #Increase the image size
            mask = np.concatenate((np.zeros([strRad+2,mask.shape[1],mask.shape[2]],dtype = np.uint8), \
                              mask, np.zeros([strRad+2,mask.shape[1],mask.shape[2]],dtype = np.uint8)), axis=0)
            mask = np.concatenate((np.zeros([mask.shape[0],strRad+2,mask.shape[2]],dtype = np.uint8), \
                              mask, np.zeros([mask.shape[0],strRad+2,mask.shape[2]],dtype = np.uint8)), axis=1)
            mask = np.concatenate((np.zeros([mask.shape[0],mask.shape[1],strRad+2],dtype = np.uint8), \
                              mask, np.zeros([mask.shape[0],mask.shape[1],strRad+2],dtype = np.uint8)), axis=2)

      elif mode == 'inverse':
            #Decrease the image size (complementary operation)
            mask = mask[strRad+2:-(strRad+2),strRad+2:-(strRad+2),strRad+2:-(strRad+2)]

      return mask
    
    def saveMasks(self,kidneyMask,vesselsMaskPrev,vesselsMask,resultPathKidney,resultPathVessels):
      """
      Save the kidney and vessels masks
      """
      fileNames = glob(f'{self.inputFolder}/*')

      if not os.path.exists(resultPathKidney):
            os.makedirs(resultPathKidney)

      '''
      if not os.path.exists(resultPathKidney2):
            os.makedirs(resultPathKidney2)
      '''

      if not os.path.exists(resultPathVessels):
            os.makedirs(resultPathVessels)
      
      if not os.path.exists(f'{resultPathVessels}-prev'):
            os.makedirs(f'{resultPathVessels}-prev')

      for i in range(kidneyMask.shape[0]):

            tif.imwrite(f'{resultPathKidney}/{os.path.basename(fileNames[i])}',kidneyMask[i,:,:])

            #tif.imwrite(f'{resultPathKidney2}/{os.path.basename(fileNames[i])}',kidneyMask2[i,:,:])

            tif.imwrite(f'{resultPathVessels}-prev/{os.path.basename(fileNames[i])}',vesselsMaskPrev[i,:,:])

            tif.imwrite(f'{resultPathVessels}/{os.path.basename(fileNames[i])}',vesselsMask[i,:,:])

      #tif.imwrite(resultPathKidney,kidneyMask)
      #tif.imwrite(resultPathKidney2,kidneyMask2)
      #tif.imwrite(resultPathVessels,vesselsMask)



    def processArraySliceBySlice(self,im = None,mask = None,thresh = None,direction = 'up',largestCC = None):
      '''
      Function to process slice by slice either an image or a mask. Only one of the two must be provided.

      If an image is provided, a threshold is required to threshold upwards or downwards slice by slice, 
      creating a mask.
      
      With this thresholded mask or, alternatively, the mask provided by the user, the the largest 
      connected component can be kept or removed slice by slice.
      '''

      '''
      # Initialize an empty array to store the modified data
      if im is not None:
            modified_array = np.zeros_like(im,dtype = 'uint8')
            numSlices = im.shape[0]
      elif mask is not None:
            modified_array = np.zeros_like(mask,dtype = 'uint8')
            numSlices = mask.shape[0]
      '''

      numSlices = im.shape[0]

      # Iterate through each slice along the first dimension
      for i in tqdm(range(numSlices)):

            if im is not None:
                  #Extract the slice and delete it from the original image
                  maskSlice = im[0, :, :]
                  if im.shape[0] > 1:         
                        im = im[1:, :, :]
                  else:
                        del im
            
                  # Threshold the slice from thresh up or from thresh down
                  if direction == 'up':
                        maskSlice = 1*(maskSlice >= thresh)
                  elif direction == 'down':
                        maskSlice = 1*(maskSlice < thresh)

                  #maskSlice = maskSlice.astype('uint8')   
                  maskSlice = maskSlice.astype('uint8')

            elif mask is not None:
                  maskSlice = mask[i, :, :]

            
            # Label the connected components in the slice
            maskSlice, num_features = label(maskSlice)
            
            # Skip if there are no components (i.e., the slice is empty)
            if num_features == 0:
                  continue
            
            # Find the largest component
            assert( maskSlice.max() != 0 ) # assume at least 1 CC
            largest_component = np.argmax(np.bincount(maskSlice.flat)[1:])+1 

            #sizes = ndi_sum(maskSlice, labeled_slice, range(1, num_features + 1))
            #largest_component = np.argmax(sizes) + 1  # +1 because component labels start at 1
            
            #Remove or keep the largest connected component
            #modified_slice = maskSlice.copy()
            #modified_slice = np.zeros_like(mask_slice)

            if largestCC == 'remove':
                  maskSlice[maskSlice == largest_component] = 0
                  maskSlice[maskSlice != 0] = 1

                  
            elif largestCC == 'keep':
                  maskSlice[maskSlice != largest_component] = 0
                  maskSlice[maskSlice!=0] = 1
            
            #modified_array[i, :, :] = modified_slice
            #modified_slice = modified_slice.astype('uint8')
            maskSlice = maskSlice.astype('uint8')

            #Store the resulting mask in a newly created 3D array
            if i ==0:
                  mask3D = maskSlice[np.newaxis,:,:]
            else:
                  mask3D = np.concatenate((mask3D,maskSlice[np.newaxis,:,:]),axis=0)                
      
      return mask3D  

    def get_kidney_and_vessels_masks(self):
      """
      Perform the whole thresholding and morphological operations pipeline.
      """

      #Define the variables from "self"
      strRad = self.strRad
      inputFolder = self.inputFolder
      #resultPathKidney = self.resultFolder+'/kidneyMask-{}_pygorpho_strRad_{}.tif'.format(os.path.basename(inputFolder),strRad)
      resultPathKidney = self.resultFolder+'/kidneyMask-{}_pygorpho_strRad_{}'.format(os.path.basename(inputFolder),strRad)
      #resultPathVessels = self.resultFolder + '/vesselsMask-{}_pygorpho_strRad_{}.tif'.format(os.path.basename(inputFolder), strRad)
      resultPathVessels = self.resultFolder + '/vesselsMask-{}_pygorpho_strRad_{}'.format(os.path.basename(inputFolder), strRad)
      kidneyThres = self.kidneyThres
      vessThres = self.vessThres
      sliceToVis = self.sliceToVis

      sliceBySliceThresholds = self.sliceBySliceThresholds
      sliceBySliceErodeKidneyMask = self.sliceBySliceErodeKidneyMask
      radiusKidneyErosion = self.radiusKidneyErosion

      fileNames = glob(f'{self.inputFolder}/*')
      print(f'{resultPathKidney}/{os.path.basename(fileNames[0])}')

      #Create structuring element for morphological operations
      strucEl = 1*(rg.sphere(2*(strRad)+1, strRad))

      #Set initial time for the duration of the whole function
      time1 = time.time()

      #Read the original image's 2D slices to create a 3D matrix
      initTime = time.time()
      origIm = tif.imread(inputFolder+'/*')
      elapsedTime = time.time()-initTime
      print('Time to load the image: {} min'.format(np.round(elapsedTime/60,3)))

      #Perform thresholding slice by slice
      #to obtain both the full kidney and vessels masks
      initTime = time.time()

      kidneyMask = self.processArraySliceBySlice(im = origIm,thresh = kidneyThres,direction = 'up',\
                                    largestCC = 'keep')

      vesselsMask = self.processArraySliceBySlice(im = origIm,thresh = vessThres,direction = 'down',\
                                                  largestCC = 'remove')
      
      #kidneyMask,vesselsMask = self.kidneyVesselsMasks(origIm, sliceBySliceThresholds,kidneyThres, vessThres)          
      elapsedTime = time.time()-initTime
      print('Time to threshold the image and obtain full kidney and vessels masks: {} min'.format(np.round(elapsedTime/60,3)))

      #Remove origIm from memory
      del origIm

      #Plot the initial kidney mask
      
      plt.figure()
      plt.imshow(kidneyMask[:,:,sliceToVis])
      plt.title('Initial full kidney mask')
      plt.show(block=False)
      

      #Pad the full kidney mask in all direction so that strRad fits in the borders of the kidney
      initTime = time.time()
      kidneyMask = self.padStrRad(kidneyMask,strRad,'direct')
      elapsedTime = time.time()-initTime
      print('Time to pad the initial full kidney mask: {} min'.format(np.round(elapsedTime/60,3)))

      #Apply pygorpho 3D morphological operations to it:

      #Opening operation
      initTime = time.time()
      kidneyMask = pg.flat.open(kidneyMask, strucEl)
      elapsedTime = time.time()-initTime
      print('Time to apply opening to the full kidney mask: {} min'.format(np.round(elapsedTime/60,3)))

      #Closing operation
      initTime = time.time()
      kidneyMask = pg.flat.close(kidneyMask, strucEl)
      elapsedTime = time.time()-initTime
      print('Time to apply closing to the full kidney mask: {} min'.format(np.round(elapsedTime/60,3)))                             

      print('kidneyMask.shape before reduction: ',kidneyMask.shape)
      initTime = time.time()
      kidneyMask = self.padStrRad(kidneyMask,strRad,'inverse')
      elapsedTime = time.time()-initTime
      print('Time to return to the original size: {} min'.format(np.round(elapsedTime/60,3)))

      print('kidneyMask.shape after reduction: ',kidneyMask.shape)
      print('vesselsMask.shape: ',vesselsMask.shape)

      
      #Uncomment this to visualize a slice of the modified mask
      plt.figure()
      plt.imshow(kidneyMask[:,:,sliceToVis])
      plt.title('Modified kidney mask after opening and closing')
      plt.show(block=False)
      

      

      #Remove the largest connected component slice by slice from the vessels mask
      vesselsMaskPrev = vesselsMask.copy()
      vesselsMask = self.processArraySliceBySlice(mask = vesselsMask, largestCC = 'remove')
            
      initTime = time.time()  


      #kidneyMask2,vesselsMask = self.combineKidneyVesselsMasks(sliceBySliceErodeKidneyMask,kidneyMask,radiusKidneyErosion,vesselsMask)
      
      elapsedTime = time.time()-initTime
      print('Time to remove the largest connected component slice by slice from the vessels mask: {} min'.format(np.round(elapsedTime/60,3)))

      
      
      #Uncomment this to visualize a slice of the modified mask
      plt.figure()
      plt.imshow(vesselsMask[:,:,sliceToVis])
      plt.title('Final vessels and gaps mask')
      plt.show()
      
      resultPathKidney2 = self.resultFolder+'/kidneyMask2-{}_pygorpho_strRad_{}'.format(os.path.basename(inputFolder),strRad)
      #resultPathKidney2 = self.resultFolder+'/kidneyMask2-{}_pygorpho_strRad_{}.tif'.format(os.path.basename(inputFolder),strRad)

      #Save the generated masks
      self.saveMasks(kidneyMask,vesselsMaskPrev,vesselsMask,resultPathKidney,resultPathKidney2,resultPathVessels)
      
      #Print the total time to run the whole script
      timeWholeScript = time.time()-time1
      print('Time to run the whole function: {}'.format(np.round(timeWholeScript/60,3)))
      
      return kidneyMask, vesselsMask
    
    def get_vessels_masks(self):
            
      '''
      Get the vessels mask from the original image slice by slice
      '''
      
      #Define the variables from "self"
      strRad = self.strRad
      inputFolder = self.inputFolder
      resultPathVessels = self.resultFolder + '/vesselsMask-{}_pygorpho_strRad_{}'.format(os.path.basename(inputFolder), strRad)
     
      vessThres = self.vessThres
      sliceToVis = self.sliceToVis

      fileNames = glob(f'{self.inputFolder}/*')

      origIm = tif.imread(inputFolder+'/*')

      #Slice by slice thresholding to get the vessels mask, removing the largest connected component
      #from each slice      
      vesselsMask = self.processArraySliceBySlice(im = origIm,thresh = vessThres,direction = 'down',\
                                                  largestCC = 'remove')

      plt.imshow(vesselsMask[sliceToVis,:,:])

      if not os.path.exists(resultPathVessels):
        os.makedirs(resultPathVessels)

      for i in range(vesselsMask.shape[0]):
        tif.imwrite(f'{resultPathVessels}/{os.path.basename(fileNames[i])}',vesselsMask[i,:,:])

    def get_kidney_masks(self):
          
      '''
      Get the kidney mask from the original image slice by slice and apply 3D morphological operations to it
      '''

      #Define the variables from "self"
      strRad = self.strRad
      inputFolder = self.inputFolder
      resultPathKidney = self.resultFolder+'/kidneyMask-{}_pygorpho_strRad_{}'.format(os.path.basename(inputFolder),strRad)
      kidneyThres = self.kidneyThres
      sliceToVis = self.sliceToVis
      processDimension = self.processDimension

      fileNames = glob(f'{self.inputFolder}/*')

      origIm = tif.imread(inputFolder+'/*')
      
      #Create the folder to save the resulting kidney mask
      if not os.path.exists(resultPathKidney):
            os.makedirs(resultPathKidney)

      #Slice by slice thresholding to get the full kidney mask, keeping the largest connected component
      #from each slice
      kidneyMask = self.processArraySliceBySlice(im = origIm,thresh = kidneyThres,direction = 'up',\
                                                  largestCC = 'keep')

      plt.figure()
      plt.imshow(kidneyMask[sliceToVis,:,:])
      plt.title('Full kidney mask before morphological operations')

      #Pad the full kidney mask in all direction so that strRad fits in the borders of the kidney
      initTime = time.time()
      kidneyMask = self.padStrRad(kidneyMask,strRad,'direct')
      elapsedTime = time.time()-initTime
      print('Time to pad the initial full kidney mask: {} min'.format(np.round(elapsedTime/60,3)))


      #Apply pygorpho 3D morphological operations to it:
      if processDimension == '3D':
            #Create spherical structuring element
            strucEl = 1*(rg.sphere(2*(strRad)+1, strRad))

            #Opening operation
            initTime = time.time()
            kidneyMask = pg.flat.open(kidneyMask, strucEl)
            elapsedTime = time.time()-initTime
            print('Time to apply 3D opening to the full kidney mask: {} min'.format(np.round(elapsedTime/60,3)))

            #Closing operation
            initTime = time.time()
            kidneyMask = pg.flat.close(kidneyMask, strucEl)
            elapsedTime = time.time()-initTime
            print('Time to apply 3D closing to the full kidney mask: {} min'.format(np.round(elapsedTime/60,3)))

      #Alternatively, apply the same operations slice by slice with a disk structuring element
      elif processDimension == '2D':                             
            #Create disk structuring element
            strucEl = 1*(rg.circle(2*(strRad)+1, strRad))

            initTime = time.time()
            for i in tqdm(range(kidneyMask.shape[0])):
                  kidneyMask[i,:,:] = pg.flat.open(kidneyMask[i,:,:], strucEl)                
            print('Time to apply 2D opening to the full kidney mask: {} min'.format(np.round(elapsedTime/60,3)))

            initTime = time.time()
            for i in tqdm(range(kidneyMask.shape[0])):
                  kidneyMask[i,:,:] = pg.flat.close(kidneyMask[i,:,:], strucEl)
            print('Time to apply 2D closing to the full kidney mask: {} min'.format(np.round(elapsedTime/60,3)))

      #Return to the original size
      print('kidneyMask.shape before reduction: ',kidneyMask.shape)
      initTime = time.time()
      kidneyMask = self.padStrRad(kidneyMask,strRad,'inverse')
      elapsedTime = time.time()-initTime
      print('Time to return to the original size: {} min'.format(np.round(elapsedTime/60,3)))

      if processDimension == '3D':
            for i in range(kidneyMask.shape[0]):
                  tif.imwrite(f'{resultPathKidney}/{os.path.basename(fileNames[i])}',kidneyMask[i,:,:])
      elif processDimension == '2D':
            for i in range(kidneyMask.shape[0]):
                  mask2D = kidneyMask[i,:,:].astype('uint8')
                  tif.imwrite(f'{resultPathKidney}/{os.path.basename(fileNames[i])}',kidneyMask[i,:,:])

      

if __name__ == '__main__':
      """
      If the script is run directly
      """
      parser = argparse.ArgumentParser()

      parser.add_argument('--strRad', type = int, default = 20,\
                  help='Folder containing all 2D slices that form the original 3D image')

      parser.add_argument('--inputFolder', type = str, default = 'E:/AAV para enfermedades renales/LSFM combined images/Full images/MacroSPIM2/R1CLeft2021_0.5_Lectine',\
                  help='Folder containing all 2D slices that form the original 3D image')

      parser.add_argument('--resultFolder', type = str, default = 'E:/Github repositories/LSFM-processing-data/Full kidney and vessels segmentation/MacroSPIM2',\
                  help='Path to save the resulting full kidney mask')

      parser.add_argument('--kidneyThres',type = int,default = 260,help='Threshold above which to detect the full kidney')

      parser.add_argument('--vessThres',type = int,default = 900,help='Threshold below which to detect vesels')

      parser.add_argument('--sliceToVis',type = int,default = 400,help='Slice to visualize')

      parser.add_argument('--sliceBySliceThresholds',type = bool,default = False,help='Run the thresholder slice by slice or at once')

      parser.add_argument('--sliceBySliceErodeKidneyMask',type = bool,default = True,help='Run the fullkidney mask erosion slice by slice or at once')

      parser.add_argument('--radiusKidneyErosion',type = int,default = 100,help='Radius of the structuring element for kidney mask erosion before multiplying with the vessels mask')
      
      parser.add_argument('--skeletonize',type = bool,default = False,help='Skeletonize the image or not')

      parser.add_argument('--processDimension',type = str,default = '3D',help='Process the image in 2D or 3D')

      args = parser.parse_args()

      processer = process_fullKidneyMasks(args)

      processer.get_kidney_and_vessels_masks()


