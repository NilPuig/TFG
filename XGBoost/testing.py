import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

model=pickle.load(open("model.dat", "rb"))

testData=pd.read_csv("cat_star_04.csv")

# Saving the ID for future uses
ID=testData['id']

basename='flux_nb'
initialnumber=455;
requiredColumns=[]
for i in range(40):
    requiredColumns.append(basename+str(initialnumber+10*i))
    

newData=pd.DataFrame()

for i in requiredColumns:
    newData[i]=testData[i]


dtest=xgb.DMatrix(data=newData)

probs=model.predict(dtest)

predictions = [1 if x > 0.50 else 0 for x in probs]

results=pd.DataFrame()
results['id']=ID
results['Predictions']=predictions

results.to_csv("Predictions.csv",index=False)

print("Predictions Complete ")

