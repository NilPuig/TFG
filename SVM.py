# -*- coding: utf-8 -*-


# Author: Nil Puig


import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def main():
    #TRN = Training
    #Val = Validation
    
    #..............get data from files.................
    file0 = 'cat_gal_03.csv'
    ObjectsDataI = np.genfromtxt(file0, comments = '#', delimiter = ',')
    file1 = 'cat_star_03.csv'
    ObjectsDataII = np.genfromtxt(file1, comments = '#', delimiter = ',')
    #print ("\n------ ObjectsDataI.shape ------ \n ")
    #print (ObjectsDataI.shape)
    print (ObjectsDataI.shape)
    print (ObjectsDataII.shape)
    #MAIN
    nStarsTRN = 5000
    nGalTRN = 7000
    nStarsVal = 1000
    nGalVal = 1000
    nfilters = 40
    firstFilter = 0
    
    # Copy 7k gal with 40 filters for TRN 
    FluxesGalTRN = np.copy(ObjectsDataI[1:nGalTRN+1,4+firstFilter:4+nfilters]) 
    # Copy 1k gal with 40 filters for VAL
    FluxesGalVal = np.copy(ObjectsDataI[nGalTRN+1: nGalTRN + 1 + nGalVal ,4+firstFilter:4+nfilters]) 

    # Copy 5k stars and 40 filters for TRN 
    FluxesStarTRN = np.copy(ObjectsDataII[1:nStarsTRN+1 , 4+firstFilter:4+nfilters]) 
    # Copy 1k stars for VAL 
    FluxesStarVal = np.copy(ObjectsDataII[nStarsTRN: nStarsTRN + nStarsVal,4+firstFilter:4+nfilters]) 
        
    # Create array to label the data as Galaxy (number 0)
    label_Gal_TRN = np.empty(nGalTRN)
    label_Gal_TRN.fill(0)
    label_Gal_VAL = np.empty(nGalVal)
    label_Gal_VAL.fill(0)
    
    # Create array to label the data as Star (number 1)
    label_Stars_TRN = np.empty(nStarsTRN)
    label_Stars_TRN.fill(1)
    label_Stars_VAL = np.empty(nStarsVal)
    label_Stars_VAL.fill(1)
    
    # 
    Gal_classTRN = np.copy(ObjectsDataI[1:nGalTRN+1,85])
    Gal_classVal = np.copy(ObjectsDataI[nGalTRN+1: nGalTRN + 1 + nGalVal,85])
    Star_classTRN = np.copy(ObjectsDataII[1:nStarsTRN+1,85])
    Star_classVal = np.copy(ObjectsDataII[nStarsTRN: nStarsTRN + nStarsVal,85])
    
    # Join fluxes of gal and stars for TRN (12k fluxes with 40 filters)
    FluxesTRN = np.concatenate((FluxesGalTRN,FluxesStarTRN), axis = 0)
    # Join labels of gal and stars for TRN
    label_TRN = np.concatenate((label_Gal_TRN,label_Stars_TRN), axis = 0)

     # Join fluxes of gal and stars for VAL (2k fluxes with 40 filters)
    FluxesVal = np.concatenate((FluxesGalVal,FluxesStarVal), axis = 0)
    # Join labels of gal and stars for VAL
    label_VAL = np.concatenate((label_Gal_VAL,label_Stars_VAL), axis = 0)
    
    print ("\n------ FluxesTRN.shape ------ \n ")
    print (FluxesTRN.shape)
    
    print (FluxesTRN[0])
    
  
    from sklearn import datasets
    iris = datasets.load_iris()
    
    from sklearn.naive_bayes import GaussianNB
    
    gnb = GaussianNB()
    gnb.fit(FluxesTRN, label_TRN)
    pred = gnb.predict(FluxesVal)
    
    from sklearn.metrics import accuracy_score
    
    print (accuracy_score(pred,label_VAL)) # around 51%
    


main()

