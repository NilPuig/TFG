
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

# Reading the data using pandas 
galaxy=pd.read_csv("cat_gal_04.csv")
star=pd.read_csv("cat_star_04.csv")

# Making a list of the required Columns
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

# Seperating out the testing data before doing upsampling
testStar=starTraining.tail(500)
testGalaxy=galaxyTraining.tail(2000)
starTraining=starTraining.head(5786)
galaxyTraining=galaxyTraining.head(29654)



# Upsampling . Please note that stars data was considered 4 times as there was an inherrent bias within the data distribution
frames=[starTraining,galaxyTraining,starTraining,starTraining,starTraining]
combined=pd.concat(frames)

# Randomizing the order of the training data so as to ensure a smooth training process
combined=combined.sample(frac=1)
test=pd.concat([testStar,testGalaxy])


# storing the labels in a different array and removing the labels from the training data
target=combined['class']
test_target=test['class']

del combined['class']
del test['class']

X_test=test.values
y_test=test.values

X=combined.values
y=target.values


# Preprocessing the given data so as to ensure a smooth training process
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)


# Defining the model Architecture . The Input layer has 40 units followed by two hidden units having 100 and 30 hidden units each. All the layers are 
# fully connected and relu is used as the activation function.
classifier = Sequential()
classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu', input_dim = 40))
classifier.add(Dense(output_dim = 30, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



validationAUC=[]
trainAUC=[]


# Writing a call back function for calculating the AUC value after each epoch and storing them in a list
class roc_callback(keras.callbacks.Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        validationAUC.append(roc_val)
        trainAUC.append(roc)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


# Training the classifier
history=classifier.fit(X, y,validation_data=(X_test, test_target), callbacks=[roc_callback(training_data=(X, y),validation_data=(X_test, test_target))],epochs=1000)


# Plotting accuracy vs epoch for the given data

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['val_acc'])
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Plotting AUC vs epoch
plt.plot(validationAUC)
plt.plot(trainAUC)
plt.title('Model AUC vs epochs')
plt.ylabel('AUC')
plt.xlabel('epoch')
plt.legend(['validation', 'train'], loc='upper left')
plt.show()


# Making predictions on Hidden data set
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

print("Final testing Accuracy obtained on the testnig data set "+str(accuracy_score(test_target,y_pred)))

# Saving the model weights into a file

model_json = classifier.to_json()
with open("modelWithUpsampling.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("modelWithUpsampling.h5")
print("Saved model to disk")


