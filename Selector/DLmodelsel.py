from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import Input,Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.random import set_seed
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from numpy.random import seed

import pandas as pd
import numpy as np
import pickle
import os,sys

import auxiliary as aux #self-created library

#Use seed for reproducibility of results
seedval = 2018
seed(seedval)
set_seed(seedval)


#load the data and prepare X and y
Data = pd.read_csv("mlp_multilabel_train_data.csv",header=None)
print('Dimension of Data ( Instances: ',Data.shape[0],', Features: ',Data.shape[1]-1,' )')

#X = Data.drop([Data.columns[-1]], axis = 1)
#y = Data[Data.columns[-1]]
X = Data.iloc[:,:-5]
y = Data.iloc[:,-4]
print('Dimension of X: ',X.shape)
print('Dimension of y: ',y.shape)

#select the train and validation set from Data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)


#load the holdout/test set
Data_test = pd.read_csv("mlp_multilabel_test_data.csv",header=None)
print('\nDimension of Data_test ( Instances: ',Data_test.shape[0],', Features: ',Data_test.shape[1]-1,' )')

#X_test = Data_test.drop([Data_test.columns[-1]], axis = 1)
#y_test = Data_test[Data_test.columns[-1]]
X_test = Data_test.iloc[:,:-5]
y_test = Data_test.iloc[:,-4]
print('Dimension of X: ',X_test.shape)
print('Dimension of y: ',y_test.shape)


#do auxilliary directory cleanup
os.system('rm -rf models')
os.system('mkdir models')


#determine the classweights
classweights = class_weight.compute_class_weight(class_weight='balanced',
                                                 classes=np.unique(y_train),
                                                 y=y_train)
class_weights = {0:classweights[0], 1:classweights[1]}
print(class_weights)

#hidden neuron grid
neuron_grid = [5]

best_score = np.inf
best_model = None

os.system('rm models/*')

#set the optimiser
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

#create all the models
aux.create_models(X_train,neuron_grid,activation='relu',dropout_rate=0.5)

#structure selection
modelcount = 0
for modelfile in os.listdir('models'):
    modelcount = modelcount+1
    print('Model Count:',modelcount)
    fp_json = open('models/'+modelfile)
    model = fp_json.read()
    model = model_from_json(model)
    model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])
    chkpointpath = 'chkpt-'+os.path.splitext(modelfile)[0]
    cb = [EarlyStopping(monitor='val_accuracy', patience=50, mode='max', min_delta=0.0001),
          ModelCheckpoint(chkpointpath, monitor='val_accuracy', save_best_only=True, mode='max')]

    #train and validate
    model.fit(X_train, y_train, 
              batch_size=32,
              validation_data=(X_val, y_val),
              class_weight=class_weights,
              epochs=1000,
              verbose=0,
              callbacks=[cb])
            
    #load the saved model during training (=best model on validation set)
    model = load_model(chkpointpath)
    val_loss = model.evaluate(X_val, y_val)[0]
    print('Val loss of the present model: ',val_loss)

    #check if this model can be selected (using validation loss; can be changed to val acc)
    if val_loss < best_score:
        best_score = val_loss
        best_model = model
        print('[Model Selected] Best val score: ',best_score,', Best model: ',modelfile)

    #delete the checkpoints for the present model (not required anymore)
    os.system('rm -rf '+chkpointpath)


#evaluate the obtained best model
y_pred_prob = best_model.predict(X_test)
y_pred = np.round_(y_pred_prob)
np.savetxt('mlp_pred.csv', y_pred, fmt='%d')

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test,y_pred))

score = best_model.evaluate(X_test,y_test)
print('Independent Test Accuracy: {}\n'.format(score[1]))

with open('score.txt', 'w') as f:
  f.write('{}\n'.format(score[1]))
f.close()

#save the best model
best_model.save('savedmodel')

#print the structure of the best_model
print(best_model.summary())


#Load the best model saved (this is just to check, if it works ok)
#best_model = load_model('savedmodel')
#score = best_model.evaluate(X_test, y_test, verbose=1)
#print('Loss:', score[0])
#print('Accuracy:', score[1])
