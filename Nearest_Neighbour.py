# -*- coding: utf-8 -*-
"""
Author: Nil Puig

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def dist_loop(x_train,x_test):
    
    """ 
    "distance_matrix" is the Euclidian distance matrix between the training 
    and the test set. 
    """
    distance_matrix = np.empty((len(x_train),len(x_test)))
    
    for x in range(len(x_train)):
        for y in range(len(x_test)):
            x_all=zip(x_train[x],x_test[y])
            a=0
            for z in range(40):
                a+=np.square(x_all[z][0]-x_all[z][1])
            distance_matrix[x][y]=np.sqrt(a)
       
    return distance_matrix

def dist_vec(x_train,x_test):
    distance_matrix = np.empty((len(x_train),len(x_test)))
    # Even though this part was intented to do without a "for" loop, it has
    # been necessary to introduce at least the following one. However, the
    # performance hasn't been affected (The whole function only takes 0.21s)
    for x in range(len(x_train)):
        distance_matrix[x]=np.sqrt(np.sum(np.square(x_train[x]-x_test),axis=1))
    return distance_matrix

def nearest_neighbor_k(distance_matrix,k,x_train_1,x_test_1,y_train_1,y_test_1):
    d,mislabeled_stars,mislabeled_gal,y_pred=0,0,0,[]
    
    # the distance matrix is transposed:
    distance_matrix=distance_matrix.transpose()
    
    # Iterate thtough all test objects
    for x in range(len(x_test_1)):
        a=0 # used to decide which label a test point belongs to
        
        #Iterate k times to find the k nearest neighbours
        for y in range(k):
            if y_train_1[np.argsort(distance_matrix[x])[0]]==2:
                a+=1
            else:
                a=a-1
        if a>0:  
            y_pred.append(2)
        else:
            y_pred.append(5)
        if y_test_1[x]==2:
            if y_pred[x]!=2:
                mislabeled_gal += 1
                d+=1
        else:
            if y_pred[x]!=5:
                mislabeled_stars += 1
                d+=1
    plural=''
    if k>1: plural='s'
    print ("\n%d - Nearest neighbor%s classifier: " 
        %(k,plural))
    print ("    Number of mislabeled points : %d, percentage error: %f "
        %(d,100*d/len(y_pred)))
    print ("    Number of mislabeled 'galaxies' : %d"%mislabeled_gal)
    print ("    Number of mislabeled 'stars' : %d"%mislabeled_stars)
    
    print "\nTest - Accuracy :", metrics.accuracy_score(y_test_1, y_pred)
    print "\nTest - classification report :\n", metrics.classification_report(y_test_1, y_pred)


    
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
    nGalTRN = 5000
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
        
    a=0
    for i in [0,3,4,5,6,7,9,10,12,13,14,15,16,19,20,21,22,23,24,26,27,28,29,30,32,35,36,39]:
        FluxesGalTRN = np.delete(FluxesGalTRN,(i-a), axis=1)
        FluxesGalVal = np.delete(FluxesGalVal,(i-a), axis=1)
        FluxesStarTRN = np.delete(FluxesStarTRN,(i-a), axis=1)
        FluxesStarVal = np.delete(FluxesStarVal,(i-a), axis=1)
        a+=1
        
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
    
    # Join fluxes of gal and stars for TRN (12k fluxes with 40 filters)
    FluxesTRN = np.concatenate((FluxesGalTRN,FluxesStarTRN), axis = 0)
    # Join labels of gal and stars for TRN
    label_TRN = np.concatenate((label_Gal_TRN,label_Stars_TRN), axis = 0)
    
    print ("\n------ FluxesTRN.shape ------ \n ")
    #print (FluxesTRN.shape)

     # Join fluxes of gal and stars for VAL (2k fluxes with 40 filters)
    FluxesVal = np.concatenate((FluxesGalVal,FluxesStarVal), axis = 0)
    # Join labels of gal and stars for VAL
    label_VAL = np.concatenate((label_Gal_VAL,label_Stars_VAL), axis = 0)
     
    
    # The distance matrix is computed using vectorization:    
    distance_matrix = dist_vec(FluxesTRN, FluxesVal)
    
    
    k = 1  # number of nearest neighbours to be found
    
    # Here, a k-nearest neighbors classifier is implemented. It uses the 
    # previously distance matrix calculated.
    nearest_neighbor_k(distance_matrix,k,FluxesTRN, FluxesVal,label_TRN,label_VAL)
    

main()
