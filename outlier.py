#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 20:40:02 2018

@author: nilpuig
"""

# -*- coding: utf-8 -*-


# Author: Nil Puig


import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def main():
    #TRN = Training
    #Val = Validation
    
    #..............get data from files.................
    file0 = 'cat_gal_04.csv'
    ObjectsDataI = np.genfromtxt(file0, comments = '#', delimiter = ',')
    file1 = 'cat_star_04.csv'
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
    
    nobjectsTRN = nGalTRN + nStarsTRN
    nobjectsVal = nStarsVal + nGalVal
    
    # Copy 7k gal with 40 filters for TRN 
    FluxesGalTRN = np.copy(ObjectsDataI[1:nGalTRN+1,4+firstFilter:4+nfilters]) 
    # Copy 1k gal with 40 filters for VAL
    FluxesGalVal = np.copy(ObjectsDataI[nGalTRN+1: nGalTRN + 1 + nGalVal ,4+firstFilter:4+nfilters]) 

    # Copy 5k stars and 40 filters for TRN 
    FluxesStarTRN = np.copy(ObjectsDataII[1:nStarsTRN+1 , 4+firstFilter:4+nfilters]) 
    # Copy 1k stars for VAL 
    FluxesStarVal = np.copy(ObjectsDataII[nStarsTRN: nStarsTRN + nStarsVal,4+firstFilter:4+nfilters]) 
        
    # Create array to label the data as Galaxy (number 2)
    label_Gal_TRN = np.empty(nGalTRN)
    label_Gal_TRN.fill(2)
    label_Gal_VAL = np.empty(nGalVal)
    label_Gal_VAL.fill(2)
    
    # Create array to label the data as Star (number 5)
    label_Stars_TRN = np.empty(nStarsTRN)
    label_Stars_TRN.fill(5)
    label_Stars_VAL = np.empty(nStarsVal)
    label_Stars_VAL.fill(5)
    
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
    
   # Since our dataset is very big, the program will take some time to complete
   #(approximately 2 hours), we recommend to only run the program once and 
   #print and save the outliers array. Then when implementing an algorithm, 
   #just remove the outlier without running this function again.
   
    num_outliers = 1000 # number of outliers to be removed
    outliers = []
    
    lenght_Cleaned = len(FluxesTRN) - num_outliers # length of the cleaned dataset
    
    # Initialize cleaned datasets
    FluxesTRN_Cleaned =  np.empty((lenght_Cleaned,40))
    label_TRN_Cleaned =  np.empty((lenght_Cleaned,40))
        
    for a in range(num_outliers):
        
        outlier_index = 0  # the index of the outlier
        max_distance = 0  # the maximum distance to the nearest neighbor
        
        for i in range(len(FluxesTRN)):
            # find distance of one point to all the other ponits
            distance_matrix =(np.sqrt(np.sum(np.square(FluxesTRN[i]-FluxesTRN),axis=1)))
            distance_matrix.sort()

            nearest_distance = distance_matrix[1]

            # Check if distance to nearest neighbor is a new maximum 
            if nearest_distance > max_distance:
                
                # check if index is already in outliers array 
                index_already_added = False
                for s in outliers:
                    if s == i:
                        index_already_added = True
                        
                #if it's not there, add it
                if (not index_already_added):
                    max_distance = nearest_distance
                    outlier_index = i

        outliers.append(outlier_index)
        

    print (outliers) 
      
    # sort outlier indexs so we can iteratite through them easily 
    outliers.sort()
    
    # create a new dataset without the outliers
    count = 0
    for i in range(len(FluxesTRN)):
        if len(outliers)==0 or i!= outliers[0]: 
            FluxesTRN_Cleaned[count] = FluxesTRN[i]
            label_TRN_Cleaned[count] = label_TRN[i]
            count += 1
        else:
            del outliers[0] 

            
    print ("\n------ FluxesTRN_Cleaned shape ------ \n ")
    print FluxesTRN_Cleaned.shape
            
main()


