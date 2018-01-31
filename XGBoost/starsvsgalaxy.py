# Author ---> Nil Puig

import pandas as pd
import numpy as np
import pickle

galaxy=pd.read_csv("cat_gal_04.csv")
star=pd.read_csv("cat_star_04.csv")

# Initial Data Processing

# Making a list of required columns
basename='flux_nb'
initialnumber=455;
requiredColumns=[]
for i in range(40):
    requiredColumns.append(basename+str(initialnumber+10*i))
    


starTraining=pd.DataFrame()
galaxyTraining=pd.DataFrame()

for i in requiredColumns:
    starTraining[i]=star[i]
    galaxyTraining[i]=galaxy[i]


# Assign class labels for gal and stars
starTraining['class']=0
galaxyTraining['class']=1



# Combining the two dataframes into a single unit
frames=[starTraining,galaxyTraining]
combined=pd.concat(frames)


# Shuffling the dataframe 
combined=combined.sample(frac=1)

# Creating the testing Dataset
test=combined.tail(5000)

# Resizing the actual Dataset to eliminate the testing data
combined=combined.head(32940)


# XGBOOST Parameters
params = {
	'silent':1,
    'objective':'binary:logistic',
    'eval_metric':'error',
    'eta':0.025,
    'max_depth':6,
    'subsample':0.7,
    'colsample_bytree':0.7,
    'min_child_weight':5
    
}

# Training Target
target=combined['class']

# Testing dataset target
test_target=test['class']

del combined['class']
del test['class']

import xgboost as xgb

dtrain = xgb.DMatrix(data=combined, label = target)
dtest = xgb.DMatrix(data=test)


# Actual Training Process

rounds = 2000 # the number of iterations the training will undergo. 
bst = xgb.cv(params, dtrain, num_boost_round=rounds, early_stopping_rounds=40,nfold=5L,verbose_eval=10)

bst_train = xgb.train(params, dtrain, num_boost_round=rounds)

# Saving the model

pickle.dump(bst_train, open("model.dat", "wb"))

# Using the trained model to make predictions
predictions=bst_train.predict(dtest)

from sklearn.metrics import accuracy_score

# Converting the predicted probabilities into classes.
predictions = [1 if x > 0.50 else 0 for x in predictions]

print (accuracy_score(test_target, predictions))

print (" Training complete - Exiting")
