# -*- coding: utf-8 -*-


# Author: Nil Puig


import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import metrics


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
    firstFilter = 0
    nfilters = 40

    
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

    y = label_TRN
    X = FluxesTRN

    

    #Normalize
    X = StandardScaler().fit_transform(X)
    
    # evaluate the model by splitting into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2017)
    
    kfold = cross_validation.StratifiedKFold(y=y_train, n_folds=5, random_state=2017)
    num_trees = 5
    
    x = []
    y = []


    for i in range(20):
        # Dection Tree with 5 fold cross validation
        # lets restrict max_depth to 1 to have more impure leaves
        clf_DT = DecisionTreeClassifier(max_depth=1, random_state=2017).fit(X_train,y_train)
      
        # Using Adaptive Boosting of 100 iteration
        clf_DT_Boost = AdaBoostClassifier(base_estimator=clf_DT, n_estimators=num_trees, learning_rate=1.5, random_state=2017).fit(X_train,y_train)
        results = cross_validation.cross_val_score(clf_DT_Boost, X_train, y_train, cv=kfold)
        print "\nDecision Tree (AdaBoosting) - CV Train : %.2f" % results.mean()
        print "Decision Tree (AdaBoosting) - Train : %.2f" % metrics.accuracy_score(clf_DT_Boost.predict(X_train), y_train)
        print "Decision Tree (AdaBoosting) - Test : %.2f" % metrics.accuracy_score(clf_DT_Boost.predict(X_test), y_test)
        
        from sklearn.ensemble import GradientBoostingClassifier
    
        # Using Gradient Boosting of 100 iterations
        clf_GBT = GradientBoostingClassifier(n_estimators=num_trees, learning_rate=1.5, random_state=2017).fit(X_train, y_train)
        results = cross_validation.cross_val_score(clf_GBT, X_train, y_train, cv=kfold)
        
        print "\nGradient Boosting - CV Train : %.2f" % results.mean()
        print "Gradient Boosting - Train : %.2f" % metrics.accuracy_score(clf_GBT.predict(X_train), y_train)
        print "Gradient Boosting - Test : %.2f" % metrics.accuracy_score(clf_GBT.predict(X_test), y_test)
    
        x.append(num_trees)
        y.append(metrics.accuracy_score(clf_GBT.predict(X_train), y_train))
        num_trees += 10
    
    print (x)
    print (y)
    
    plt.scatter(x,y)
    plt.show()

main()


