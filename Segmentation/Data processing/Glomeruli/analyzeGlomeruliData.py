import matplotlib.pyplot as plt
import pandas as pd

def analyzeGlomeruliData(dataPath):
    '''This function reads a txt file containing data about specific glomeruli
    and plots those data
    '''

    data = pd.read_csv(dataPath)
    print(data)
    
    plt.figure()
    plt.plot(data['label'],data['area'],'o')
    plt.title('Glomeruli area')

    plt.figure()
    plt.plot(data['label'],data['centroid-0'],'o')
    plt.title('Glomeruli z-dimension')
