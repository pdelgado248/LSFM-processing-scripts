from glob import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import tifffile as tif 
import time 
from scipy.spatial import KDTree
from scipy.stats import shapiro
import pingouin as pg


def plotBarsAndPoints(titleTag,ylabelTag,healthyDataVoxels,pathologicalDataVoxels=None,dataset = 'healthy',\
                    voxelSize=None,ttest = False,saving = False,figuresFolder = None, dataDim = 3,yLims = None,\
                    scientNot = False):

    '''
    Plots the data in healthData and pathologicalData as bars of mean +- std with individual data points
    titleTag: str
        Title of the plot
    ylabelTag: str
        Label of the y-axis
    dataset: str
        It can be 'both' to plot healthy and pathological kidneys at the same time. Otherwise, only 'pathological' or 'healthy' kidneys are plotted
    healthyDataVoxels: list or numpy array of 1D
        Data of the healthy group. In case there are no two groups, this is the only data that will be plotted
    pathologicalDataVoxels: list or numpy array of 1D
        Data of the pathological group  
    voxelSize: float 
        Voxel size of the images (in micrometers). Only applied when it is specifically set, to resize voxels/voxel sides. Otherwise, not used.
    ttest: bool
        If True, performs a t-test between healthData and pathologicalData to check if the difference
        between their means is significant
    saving: bool    
        If True, saves the plot in the figures folder
    figuresFolder: str
        Path to the folder where the figures will be saved
    dataDim: int
        Data in 1, 2 or 3 dimensions (length, area, volume) to use voxelSize appropiately
    yLims: list
        List with the limits of the y-axis
    scientNot: bool
        Whether to use scientific notation in the y-axis. If False, it is not used
    '''
    #Position for the 2 possible sets of data in the x-axis of the plot
    position1 = 0
    position2 = 1.5

    
    #Use voxelSize (in microm) to get proper units in mm or mm^3
    if voxelSize is not None:
        #1 mm = 10^3 micrometers, so the conversion factor is 3 x the number of dimensions dataDim (1 for mm, 3 for mm^3)

        #If the data is in 2 or 3 dimensions, the voxel size must be squared or cubed
        if dataDim == 2:
            voxelSize = voxelSize**2
        elif dataDim == 3:
            voxelSize = voxelSize**3  

        factor = 3*dataDim
        healthyData = (voxelSize/(10**factor))*healthyDataVoxels
        if dataset == 'both':
            pathologicalData = (voxelSize/(10**factor))*pathologicalDataVoxels

    else:
        healthyData = healthyDataVoxels
        pathologicalData = pathologicalDataVoxels   

    # Plotting means +- std
    #For 2 datasets, the figure is bigger   
    if dataset == 'both':
        plt.figure(figsize=(4,6))
        plt.xlim(-1, 3)
    else:           
        plt.figure(figsize=(3,6))
        plt.xlim([-2, 2])  # Set x-axis start and finish

    if yLims is not None:
        plt.ylim(yLims)

    if scientNot is True:
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    barWidth = 0.25
    alphaValue = 0.5

    if dataset == 'both' or dataset == 'healthy':
        barColor = 'darkgreen'
    #If only pathological kidneys are analyzed, the color is red
    elif dataset == 'pathological':
        barColor = 'darkred'

    # Perform Shapiro-Wilk normality test on the first dataset
    shapiro_healthy_stat, shapiro_healthy_p = shapiro(healthyData)
    print(f'\n\n{titleTag}:\n')

    if dataset == 'both':
        dataName = 'Healthy'
    else:
        dataName = dataset

    print(f"Shapiro-Wilk test for {dataName} Data: p-value = {shapiro_healthy_p:.4f}")

    if shapiro_healthy_p > 0.05:
        print(f"Data can be considered normally distributed (p = {shapiro_healthy_p} > 0.05).")
    else:
        print(f"Data is NOT normally distributed (p ={shapiro_healthy_p} ≤ 0.05).")
    #Plotting first set of data (either healthy or pathological)
    #plt.bar([0], np.mean(healthyData), color=barColor, width = barWidth, alpha=alphaValue)
    violin_first_set = plt.violinplot(healthyData, positions=[position1], showmeans=True, showextrema=True)
    
    #Set adequate color
    for pc in violin_first_set['bodies']:  
        pc.set_facecolor(barColor)  # Set the color for healthy data
        pc.set_edgecolor('black')  # Set the outline color
        pc.set_alpha(alphaValue)  # Adjust transparency
    
    #Change the color of the inner lines (mean, median, extrema)
    for key in ['cmeans', 'cmedians', 'cmins', 'cmaxes', 'cbars']:
        if key in violin_first_set:
            violin_first_set[key].set_color(barColor)  # Set inner line color
            violin_first_set[key].set_linewidth(2)   # Adjust line thickness if needed


    #plt.errorbar([0], np.mean(healthyData), np.std(healthyData), fmt='.', elinewidth=2, capthick=2, errorevery=1, capsize = 10, color='black')

    #Plotting individual data points (first set of data)
    numPoints = len(healthyData)
    pointPosit = position1 + np.linspace(-0.05,0.05,numPoints)
    count=-1
    for i in range(numPoints):
        count+=1
        plt.plot(pointPosit[count],healthyData[i],'.',ms = 14,color = barColor)

    #If both datasets are analyzed, plot the second set of data
    if dataset == 'both':

        #Shapiro-Wilk normality test on the second dataset (normally pathological)
        shapiro_pathological_stat, shapiro_pathological_p = shapiro(pathologicalData)
        print(f"Shapiro-Wilk test for Pathological Data: p-value = {shapiro_pathological_p:.4f}")
        if shapiro_pathological_p > 0.05:
            print(f"Data can be considered normally distributed (p = {shapiro_pathological_p} > 0.05).")
        else:
            print(f"Data is NOT normally distributed (p = {shapiro_pathological_p} ≤ 0.05).")        
        
        
        if ttest is True:
            #Significant figures to round the p-value
            sig_figs = 2
            n1 = len(healthyDataVoxels)
            n2 = len(pathologicalDataVoxels)
            # Z value for 95% CI
            z = 1.96

            if shapiro_healthy_p > 0.05 and shapiro_pathological_p > 0.05:
                print("Both datasets are normally distributed. A t-test can be performed.\n")
                # Perform the t-test with effect size and CI
                results = pg.ttest(x=healthyDataVoxels, y=pathologicalDataVoxels, correction=False)
                p_value = results['p-val'][0]
                cohen_d = results['cohen-d'][0]
                # Calculating cohen's d confidence interval for the effect size:'
                # Pooled sample size
                dof = n1 + n2 - 2
                # Standard error of Cohen's d
                se_d = np.sqrt((n1 + n2) / (n1 * n2) + (cohen_d ** 2) / (2 * dof))
                ci_cohen_lower = cohen_d - z * se_d
                ci_cohen_upper = cohen_d + z * se_d

            else:
                print("At least one of the datasets is not normally distributed. A Mann-Witney U test should be performed.\n")
                #Perform the Mann-Witney U test with effect size and CI
                results = pg.mwu(x=healthyDataVoxels, y=pathologicalDataVoxels)
                p_value = results['p-val'][0]
                rbc= results['RBC'].iloc[0]  # Rank-biserial correlation as effect size
                se_rbc = np.sqrt((n1 + n2 + 1)/(3 * n1 * n2))  # Standard error of rank-biserial correlation
                ci_rbc_lower = rbc - z * se_rbc
                ci_rbc_upper = rbc + z * se_rbc            

        #Plotting second set of data
        #plt.bar([0.5], np.mean(pathologicalData), color='darkred', width = barWidth, alpha=alphaValue)
        violin_second_set = plt.violinplot(pathologicalData, positions=[position2], showmeans=True, showextrema=True)
        
        #Setting adequate color
        for pc in violin_second_set['bodies']:  
            pc.set_facecolor('darkred')  # Set the color for pathological data
            pc.set_edgecolor('black')  # Set the outline color
            pc.set_alpha(alphaValue)  # Adjust transparency
        
        #Change the color of the inner lines (mean, median, extrema)
        for key in ['cmeans', 'cmedians', 'cmins', 'cmaxes', 'cbars']:
            if key in violin_second_set:
                violin_second_set[key].set_color('darkred')  # Set inner line color
                violin_second_set[key].set_linewidth(2)   # Adjust line thickness if needed

        #plt.errorbar([0.5], np.mean(pathologicalData), np.std(pathologicalData), fmt='.', elinewidth=2, capthick=2, errorevery=1, capsize = 10, color='black')
        
        #Plotting individual data points (second set of data)
        numPoints = len(pathologicalData)
        pointPosit = position2 + np.linspace(-0.05,0.05,numPoints)
        count=-1
        for i in range(numPoints):
            count+=1
            plt.plot(pointPosit[count],pathologicalData[i],'.',ms = 14,color = 'darkred')

    #Plotting titles and labels

    #If a t-test is performed, write the p-value in the title and print it
    if ttest is True:
        
        #Write the p-value in the title
        if p_value < 0.01:
            # Find the nearest power of 10 ceiling for the p-value
            nearest_pow = 10**np.ceil(np.log10(p_value))
            p_value_str = f"p-value < {nearest_pow}"
        else:
            # Format normally if p-value is >= 0.01
            p_value_str = f"p-value = {p_value:.3g}"

        #Specify the type of test that was performed
        if shapiro_healthy_p > 0.05 and shapiro_pathological_p > 0.05:
            testName = 't-test'
            parameterName = 'means'
        else:
            testName = ' U-test'
            parameterName = 'medians'

        plt.title(f'{titleTag}\n({testName} {p_value_str})\n', fontsize=16)
        print(f'\n{testName} p-value: {np.format_float_scientific(p_value, precision=sig_figs)}')

        if p_value < 0.05:

            print(f"The difference between the {parameterName} is statistically significant.")

        else:
            print(f"The difference between the {parameterName} is NOT statistically significant.")

        if testName == 't-test':
            print(f"Cohen's d: {cohen_d:.4f}")
            print(f"95% CI for Cohen's d: [{ci_cohen_lower:.4f},{ci_cohen_upper:.4f}]")
        elif testName == ' U-test':
            print(f"Rank-biserial correlation: {rbc:.4f}")
            print(f"95% CI for rank-biserial correlation: [{ci_rbc_lower:.4f},{ci_rbc_upper:.4f}]")
    else:
        plt.title(f'{titleTag}\n',fontsize=16)
    

    #Label the y axis and set the ticks' size
    plt.ylabel(ylabelTag,fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)

    #Naming the x-axis  
    if dataset == 'both':
        plt.xticks([position1,position2], ['Healthy','Pathological'],fontsize=14)
    elif dataset == 'healthy':
        plt.xticks([position1], ['Healthy'],fontsize=14)
    elif dataset == 'pathological':
        plt.xticks([position1], ['Pathological'],fontsize=14)
    #Save the figure if saving is True
    if saving is True:
        plt.savefig(f'{figuresFolder}/{titleTag}.png',bbox_inches='tight')

    if dataset == 'both':
        return np.mean(healthyData),np.std(healthyData),np.mean(pathologicalData),np.std(pathologicalData)
    
    elif dataset == 'healthy' or dataset == 'pathological':
        return np.mean(healthyData),np.std(healthyData)


def readDataFromTxt(mainPath,type = 'kidney',dataset = 'one',healthyPaths=None,pathologicalPaths=None):
    '''
    Reads a series of txt files containing each of them the volume of a 
    binary mask

    mainPath: str
        Path to the main folder where the txt files are stored
    type: str
        Type of data to analyze. It can be 'kidney', 'kidneyTensorEigVals','vessels', 
        'innerRegion', 'cystsInnerRegion', 'glomeruliInnerRegion'
        
    dataset: str
        It can be 'one' to just analyze pathological or healthy kidneys. Otherwise, 'both' 
        to analyze healthy and pathological kidneys at the same time. In this case, healthyPaths
        and pathologicalPaths must be provided
    healthyPaths: list
        List of paths to the healthy kidneys
    pathologicalPaths: list
        List of paths to the pathological kidneys
    
    '''

    mainFolders = glob(f'{mainPath}/*')

    if type == 'cystsInnerRegion' or type == 'glomeruliInnerRegion':
        if dataset == 'both':
            #Initialize the vectors to store the volumes
            healthyData =  np.zeros([len(healthyPaths),2])
            countHealthy = -1

            pathologicalData = np.zeros([len(pathologicalPaths),2]) 
            countPathological = -1

        elif dataset == 'one':
            #Initialize the vectors to store the volumes
            data = np.empty((0, 2))

    elif type == 'kidneyTensorEigVals':
        if dataset == 'both':
            #Initialize the vectors to store the volumes
            healthyData =  np.zeros([len(healthyPaths),3])
            countHealthy = -1

            pathologicalData = np.zeros([len(pathologicalPaths),3]) 
            countPathological = -1

        elif dataset == 'one':
            #Initialize the vectors to store the volumes
            data = np.empty((0, 3))

    else:
        if dataset == 'both':
            #Initialize the vectors to store the volumes
            healthyData =  np.zeros(len(healthyPaths))
            countHealthy = -1

            pathologicalData = np.zeros(len(pathologicalPaths)) 
            countPathological = -1

        elif dataset == 'one':
            #Initialize the vectors to store the volumes
            data = np.array([])


    for mainFolder in mainFolders:
        
        folderName = os.path.basename(mainFolder)

        #print(folderName)

        #Get the names of the healthy kidneys that are in this main folder
        healthyPathsFolder = [os.path.basename(i) for i in healthyPaths if folderName == os.path.dirname(i)]

        #Get the names of the pathological kidneys that are in this main folder
        pathologicalPathsFolder = [os.path.basename(i) for i in pathologicalPaths if folderName == os.path.dirname(i)]
        
        #If the type is one of these 2, take specific txt files
        if type == 'glomeruliInnerRegion' or type == 'cystsInnerRegion':
            txtFilesInFolder = glob(f'{mainFolder}/*{type}.txt')
        #Otherwise, take all the txt files in the folder (they should be the same)

        elif type == 'innerRegion':
            txtFilesInFolder = glob(f'{mainFolder}/*volume.txt')
        else:
            txtFilesInFolder = glob(f'{mainFolder}/*.txt')


        #Loop over all the txt files in the folder
        for txtFile in txtFilesInFolder:
            #print(txtFile)
            #Get the name of the txt file without additional information
            if type == 'kidney':
                txtFileName = os.path.basename(txtFile).split('kidneyMask-')[1].split('-imDims-')[0]
                txtData = pd.read_csv(txtFile)
                #Calculate the volume of the kidney
                result = txtData['area'][0]

            if type == 'kidneyTensorEigVals':
                #If you want to retriecve the tensor eigenvalues
                txtFileName = os.path.basename(txtFile).split('kidneyMask-')[1].split('-imDims-')[0]
                txtData = pd.read_csv(txtFile)
                #Calculate the volume of the kidney
                result = np.array([txtData['inertia_tensor_eigvals-0'][0],txtData['inertia_tensor_eigvals-1'][0],\
                                txtData['inertia_tensor_eigvals-2'][0]])
                result = result[np.newaxis,:]

            elif type == 'vessels':
                txtFileName = os.path.basename(txtFile).split('vesselsMask-')[1].split('-volume.txt')[0]
                #Read the txt file and extract the vessels mask volume
                with open(txtFile, 'r') as f:
                    line = f.readline()
                label, value = line.split(',')
                result = int(value)  # Convert the volume to a number
            elif type == 'cystsInnerRegion' or type == 'glomeruliInnerRegion':
                txtFileName = os.path.basename(txtFile).split(f'-{type}')[0]
                txtFileName = f'{txtFileName}_pygorpho_strRad_20'
                with open(txtFile, 'r') as f:
                    # Read the header (first line)
                    header = f.readline().strip()
                     # Read the second line containing the numbers
                    data_line = f.readline().strip()
                # Split the second line by comma to get the numbers
                numCentroids, numCentroidsInside = map(int, data_line.split(','))
                #Get the ratio of centroids inside the inner region
                result = np.array([numCentroids,numCentroidsInside])
                result = result[np.newaxis,:]

            elif type == 'innerRegion':
                txtFileName = os.path.basename(txtFile).split('kidneyMask-')[1].split('-volume')[0]
                with open(txtFile, 'r') as f:
                    line = f.readline()  
                # Split the line by comma to get the numbers
                label,value = line.split(',') 
                
                result = int(value)  


            #Save the data in the corresponding vector depending on
            #if the kidney is healthy or pathological
            if dataset == 'both':

                if type == 'cystsInnerRegion' or type == 'glomeruliInnerRegion':
                    if txtFileName in healthyPathsFolder:
                        countHealthy += 1
                        healthyData[countHealthy,:] = result
                    elif txtFileName in pathologicalPathsFolder:
                        countPathological += 1
                        pathologicalData[countPathological,:] = result
                else:
                    if txtFileName in healthyPathsFolder:
                        countHealthy += 1
                        healthyData[countHealthy] = result

                    elif txtFileName in pathologicalPathsFolder:
                        countPathological += 1
                        pathologicalData[countPathological] = result

            #If only one dataset is analyzed, it doesn't matter
            elif dataset == 'one':
                if type == 'cystsInnerRegion' or type == 'glomeruliInnerRegion':
                    data = np.append(data, result,axis = 0) 
                else:
                    data = np.append(data, result)

                

    if dataset == 'both': 
        return healthyData, pathologicalData
    elif dataset == 'one':
        return data
    

def measuresFromConCompsTxt(mainPath,typeAnalysis = 'meanNearestNeighbors',typeData = 'glomeruli',\
                            healthyPaths=None,pathologicalPaths=None,percentile = 100,figuresFolder = None,\
                            saveDistr=False,voxelSize=None):
    '''
    It reads the txt files containing the connected components parameters and calculates 
    different measures.

    mainPath: str
        Path to the main folder where the centroid coordinates, volumes and other parameters of the
        connected components are stored in txt files
    typeAnalysis: str  
        Type of analysis. It can be 'meanNearestNeighbors', 'meanVolumes','showDistributions' 
    typeData: str
        Type of data to analyze. It can be 'glomeruli' or 'cysts'
    healthyPaths: list
        List of paths to the healthy kidneys
    pathologicalPaths: list
        List of paths to the pathological kidneys
    percentile: int
        Percentile to cut the histogram and not show outliers
    figuresFolder: str
        Path to the folder where the figures will be saved
    saveDistr: bool
        If True, the histograms are saved in the figures folder
    voxelSize: float
        Voxel size of the images (in micrometers). Only applied when it is specifically set, to resize voxels/voxel sides. Otherwise, not used.
    '''
    

    healthyData =  np.zeros([len(healthyPaths),3])
    countHealthy = -1

    pathologicalData = np.zeros([len(pathologicalPaths),3]) 
    countPathological = -1
    
    
    #Get the main folders
    mainFolders = glob(f'{mainPath}/*')

    countHealthyHisto = 0
    countPathologicalHisto = 0

    for mainFolder in mainFolders:
        
        folderName = os.path.basename(mainFolder)

        #Get the names of the healthy kidneys that are in this main folder
        healthyPathsFolder = [os.path.basename(i) for i in healthyPaths if folderName == os.path.dirname(i)]

        #Get the names of the pathological kidneys that are in this main folder
        pathologicalPathsFolder = [os.path.basename(i) for i in pathologicalPaths if folderName == os.path.dirname(i)]

        #Take all txt files in the folder
        txtFileNames = glob(f'{mainFolder}/*.txt')


        #Loop over all the txt files
        for i,txtFile in enumerate(txtFileNames):

            #Load the txt information as a dataframe
            data = pd.read_csv(txtFile)
            numComponents = len(data)


            if typeData == 'glomeruli':
                
                #Create the name of the txt file without extras
                txtFileName = os.path.basename(txtFile).split('_LabelledGlom')[0]
                txtFileName = f'{txtFileName}_0.5_Lectine'
                txtFileName = f'{txtFileName}_pygorpho_strRad_20'

            elif typeData == 'cysts':
                #Extract the corresponding inner mask name
                txtFileName = os.path.basename(txtFile).split('-imDims')[0]
                if 'CystsMask-' in txtFileName:
                    txtFileName = txtFileName.split('CystsMask-')[1]
                txtFileName = f'{txtFileName}_pygorpho_strRad_20'



            if typeAnalysis == 'meanNearestNeighbors':
                #Load the centroid coordinates in an array
                coords =  data[['centroid-0','centroid-1','centroid-2']].values
                
                #Find the distance to the nearest neighbor for each
                #centroid
                tree = KDTree(coords)
                distances, indices = tree.query(coords, k=2)
                nnDist = distances[:, 1]

                #Calculate the mean and std of the nearest neighbor distances
                resultMean = np.mean(nnDist)    
                resultStd = np.std(nnDist)


            elif typeAnalysis == 'meanVolumes':
                areas = data['area'].values
                resultMean = np.mean(areas)
                resultStd = np.std(areas)

            elif typeAnalysis == 'meanSurfaceArea':
                surfaceAreas = data['surface_area'].values
                resultMean = np.mean(surfaceAreas)
                resultStd = np.std(surfaceAreas)
            
            elif typeAnalysis == 'meanSphericity':
                sphericities = data['sphericity'].values
                resultMean = np.mean(sphericities)
                resultStd = np.std(sphericities)

            elif typeAnalysis == 'showDistributions': 
                areas = data['area'].values

                coords =  data[['centroid-0','centroid-1','centroid-2']].values

                #Find the distance to the nearest neighbor for each
                #centroid
                tree = KDTree(coords)
                distances, indices = tree.query(coords, k=2)
                nnDist = distances[:, 1]
                if txtFileName in healthyPathsFolder:
                    tag ='HEALTHY'
                    countHealthyHisto += 1
                    countKidneyHisto =  countHealthyHisto 
                
                elif txtFileName in pathologicalPathsFolder:
                    tag = 'PATHOLOGICAL'
                    countPathologicalHisto += 1
                    countKidneyHisto =  countPathologicalHisto

                #Rescale volumes to mm^3
                if voxelSize is not None:
                    areas = (voxelSize**3/1000**3)*areas

                #Percentile to cut the histogram and not show outliers
                percentCutArea = np.percentile(areas, percentile)

                #Plotting the histogram
                plt.figure()
                plt.hist(areas,bins=100)
                plt.autoscale(enable=True, axis='x', tight=True)
                plt.xlim(0, percentCutArea)
                plt.xlabel('mm^3')
                plt.title(f'{tag} {countKidneyHisto}\nGlomeruli volumes')

                if saveDistr is True:
                    plt.savefig(f'{figuresFolder}/{tag}_{countKidneyHisto}_{txtFileName}-glomeruliAreas.png',bbox_inches='tight')

                #Rescale nnDist to mm
                if voxelSize is not None:
                    nnDist = (voxelSize/1000)*nnDist

                #Percentile to cut the histogram and not show outliers
                percentCutnnDist = np.percentile(nnDist, percentile)

                #Plotting the histogram

                plt.figure()
                plt.hist(nnDist,bins=100)
                plt.autoscale(enable=True, axis='x', tight=True)
                plt.xlim(0, percentCutnnDist)
                plt.xlabel('mm')
                plt.title(f'{tag} {countKidneyHisto}\nGlomeruli nearest neighbor distances')

                if saveDistr is True:
                    plt.savefig(f'{figuresFolder}/{tag}_{countKidneyHisto}_{txtFileName}-glomerulinnDist.png',bbox_inches='tight')

            
            if  typeAnalysis != 'showDistributions':

                #Save the data in the corresponding vector depending on
                #if the kidney is healthy or pathological
                if txtFileName in healthyPathsFolder:
                    countHealthy += 1
                    healthyData[countHealthy,:] = [numComponents,resultMean,resultStd]
                elif txtFileName in pathologicalPathsFolder:
                    countPathological += 1
                    pathologicalData[countPathological,:] = [numComponents,resultMean,resultStd]

    if typeAnalysis != 'showDistributions':
        return healthyData, pathologicalData
    else:
        return