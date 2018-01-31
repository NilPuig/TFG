
# importing the required Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keras
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from keras.models import model_from_json


# Loading the saved model into memory
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


# Reading the data using pandas 
galaxy=pd.read_csv("cat_gal_04.csv")
star=pd.read_csv("cat_star_04.csv")

basename='flux_nb'
initialnumber=455;
requiredColumns=[]
for i in range(40):
    requiredColumns.append(basename+str(initialnumber+10*i))


# Copying the required Data alone to a new data frame
starTraining=pd.DataFrame()
galaxyTraining=pd.DataFrame()
for i in requiredColumns:
    starTraining[i]=star[i]
    galaxyTraining[i]=galaxy[i]


starTraining['class']=0
galaxyTraining['class']=1

testing=pd.concat([starTraining,galaxyTraining])
test_target=testing['class']

del testing['class']

X=testing.values
Y=test_target.values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

y_pred = loaded_model.predict(X)
y_pred = (y_pred > 0.5)


print (accuracy_score(Y,y_pred))





