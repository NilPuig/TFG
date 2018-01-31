# -*- coding: utf-8 -*-


# Author: Nil Puig


import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

def plot_decision_regions(X, y, classifier):
    
    h = .02  # step size in the mesh
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                           np.arange(x2_min, x2_max, h))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(-0.01, 0.2)
    plt.ylim(-0.01, 0.2)

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

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
    firstFilter = 34
    nfilters = 36

    
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

    
    print('Class labels:', np.unique(y))
    
    
    
    sc = StandardScaler()
    sc.fit(X)
    X = sc.transform(X)
    
    # split data into train and test
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = tree.DecisionTreeClassifier(criterion = 'entropy', random_state=0)
    clf.fit(X_train, y_train)
    
    # generate evaluation metrics
    print "Train - Accuracy :", metrics.accuracy_score(y_train, clf.predict(X_train))
    print "Train - Confusion matrix :",metrics.confusion_matrix(y_train, clf.predict(X_train))
    print "Train - classification report :", metrics.classification_report(y_train, clf.predict(X_train))
    
    print "Test - Accuracy :", metrics.accuracy_score(y_test, clf.predict(X_test))
    print "Test - Confusion matrix :",metrics.confusion_matrix(y_test, clf.predict(X_test))
    print "Test - classification report :", metrics.classification_report(y_test, clf.predict(X_test))
    plot_decision_regions(X,y,clf)
main()


