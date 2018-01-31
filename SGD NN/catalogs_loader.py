from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp
import numpy as np
import gzip
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
#type 0 gal
#type 1 star
#----------------------------------------------------------------------------
#FUNCTIONS

def load_catalogs():
    nStarsTRN = 5000
    nGalTRN = 7000
    nStarsVal = 1000
    nGalVal = 1000
    nfilters = 40
    nfilters_reduced = 40
    
    nbands_filter = int(nfilters/nfilters_reduced)
    nbands_filter = int(nfilters/nfilters_reduced)
    
    nobjectsTRN = nGalTRN + nStarsTRN
    nobjectsVal = nStarsVal + nGalVal
    
    #..............from files.................
    file0 = 'cat_gal_03.csv'
    ObjectsDataI = np.genfromtxt(file0, comments = '#', delimiter = ',')
    print ("In load_data: ObjectsDataI shape: ", ObjectsDataI.shape)
    FluxesGalTRN = np.copy(ObjectsDataI[1:nGalTRN+1,4:4+nfilters])
    Gal_classTRN = np.copy(ObjectsDataI[1:nGalTRN+1,85])
    FluxesGalVal = np.copy(ObjectsDataI[nGalTRN+1: nGalTRN + 1 + nGalVal ,4:4+nfilters])
    Gal_classVal = np.copy(ObjectsDataI[nGalTRN+1: nGalTRN + 1 + nGalVal,85])
    
    file1 = 'cat_star_03.csv'
    ObjectsDataII = np.genfromtxt(file1, comments = '#', delimiter = ',')
    print ("In filesJoiner: ObjectsDataII shape: ",ObjectsDataII.shape)
    FluxesStarTRN = np.copy(ObjectsDataII[1:nStarsTRN+1 ,4:4+nfilters])
    FluxesStarVal = np.copy(ObjectsDataII[nStarsTRN: nStarsTRN + nStarsVal,4:4+nfilters])
    Star_classTRN = np.copy(ObjectsDataII[1:nStarsTRN+1,85])
    Star_classVal = np.copy(ObjectsDataII[nStarsTRN: nStarsTRN + nStarsVal,85])
    
    FluxesTRN = np.concatenate((FluxesGalTRN,FluxesStarTRN,), axis = 0)
    Object_classTRN = np.concatenate((Gal_classTRN,Star_classTRN), axis = 0)
    
    FluxesVal = np.concatenate((FluxesGalVal,FluxesStarVal,), axis = 0)
    Object_classVal = np.concatenate((Gal_classVal,Star_classVal))
    
    FluxesTRN, Object_classTRN, nobjectsTRN = Data_cleaner(FluxesTRN, Object_classTRN, nobjectsTRN,nfilters)
    FluxesVal, Object_classVal, nobjectsVal = Data_cleaner(FluxesVal, Object_classVal, nobjectsVal,nfilters)
    FluxesTRN, Object_classTRN, nobjectsTRN = negative_corrector(FluxesTRN, Object_classTRN, ncolumns = nfilters, nrows = nobjectsTRN)
    FluxesVal, Object_classVal, nobjectsVal = negative_corrector(FluxesVal, Object_classVal, ncolumns = nfilters, nrows = nobjectsVal)
    #FluxesTRN = FluxesTRN - 0.15
    #FluxesVal = FluxesVal - 0.15
    
    Object_classTRN = np.reshape(Object_classTRN,(nobjectsTRN,1))
    Object_classVal = np.reshape(Object_classVal,(nobjectsVal,1))
    
    # The following code is to make the training_inputs and test_inputs from the information
    # in the FluxesTRN and FluesVal arrays.
    # The training_inputs and test_inputs are needed in the NN program of Michael Dobrzaski, 2016,
    # in the format in which they are created here. They need to be passed as a 2-entry tuple "training_data".
    # E. Fernandez Dec-2017
    
    # Make some tests of the star and galaxies arrays
    
    print (" Objects_classVal shape", Object_classVal.shape)
    
    print ("In load_data: nobjects for TRN and Val ", nobjectsTRN, nobjectsVal)
    
    #   This is left in case debugging is needed.
    #   for kk in range(1):   #print the first object (out of many many
    #   print (" length of Object_classTRN[0] = ", len(Object_classTRN[0]) )
    #   print ("kk = ", kk," Object_classTRN[kk] = ", Object_classTRN[kk])
    #   print ("kk = ", kk," FluxesTRN[kk] \n", FluxesTRN[kk])
    #   print ("kk = ", kk," Object_classVal[kk] = ", Object_classVal[kk])
    #   print ("kk = ", kk," FluxesVal[kk] \n", FluxesVal[kk])
    #
    n_training = nobjectsTRN
    n_testing = nobjectsVal
    
    training_inputs = FluxesTRN[0:n_training]
    training_inputs = [np.reshape(x, (40, 1)) for x in training_inputs]
    vector_TRN = [vectorized_result( int(Object_classTRN[kk][0] )) for kk in range(n_training) ]
    training_results = vector_TRN[0:n_training]
    print ( " shape training_inputs ", np.asarray(training_inputs).shape)
    print ( " shape training_results", np.asarray(training_results).shape)
    training_data = zip(training_inputs, training_results)
    
    # Call "test_data" to the validation data, for our purposes.
    
    test_inputs = FluxesVal[0:n_testing]
    test_inputs = [np.reshape(x, (40, 1)) for x in test_inputs]
    test_results = Object_classVal
    
    print ("shape of test_inputs ", np.asarray(test_inputs).shape)
    print ("shapes test_results  ", np.asarray(test_results).shape)
    test_data = zip(test_inputs, test_results)
    
    return (training_data, test_data)

def negative_corrector(data,data_class,ncolumns,nrows):
    lambdasOnPau = np.loadtxt('lambdaPlots.txt', unpack = True)
    objOut = []
    for i in range(nrows):
        out = []
        Object = data[i][:]
        [out.append(j) for j in range(ncolumns) if Object[j] <0 ]
        if len(out)<10:
            data[i,:] = np.interp(lambdasOnPau,np.delete(lambdasOnPau,out),np.delete(Object, out) )
        else:
            objOut.append(i)
    data = np.delete(data,objOut,axis = 0)
    data_class = np.delete(data_class,objOut,axis = 0)
    objects = data.shape[0]
    return data,data_class,objects

def Data_cleaner(data,data_class,objects,filters):
    out = []
    data = np.nan_to_num(data)
    [out.append(i) for i in range(objects) if LA.norm(data[i,:])== 0 ]
    data = np.delete(data,out,axis = 0 )
    data_class = np.delete(data_class,out, axis = 0)
    objects = data.shape[0]
    data = normalize(data)
    return data,data_class,objects

def vectorized_result(j):
    #    Return a 2-dimensional unit vector with a 1.0 in the jth
    #    position and zeroes elsewhere.  This is used to convert a digit
    #    (0,1) into a corresponding desired output from the neural network.
    e = np.zeros((2,1))
    e[j] = 1.0
    return e









