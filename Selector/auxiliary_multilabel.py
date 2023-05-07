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


#set seeds
seed_val = 2018
seed(seed_val)
set_seed(seed_val)



#This creates all models from 1-4 layers given the training data shape and neuron_grid for layers
def create_models(X_train, y_train, hidden_neuron_grid, activation='relu', dropout_rate=0.5):
    inputs = Input(shape=(X_train.shape[1],))
    model=Dropout(dropout_rate)(inputs)
    modelname=""
    parentmodel=model

    #First layer
    for layer1units in hidden_neuron_grid:
        model=parentmodel
        modelname=str(layer1units)
        model=Dense(layer1units, activation=activation)(model)
        model=Dropout(dropout_rate)(model)
        predictions=Dense(y_train.shape[1], activation='sigmoid')(model)
        finalmodel = Model(inputs=inputs, outputs=predictions)
        model_json = finalmodel.to_json()
        print(modelname)
        with open("models/model_"+modelname+".json", "w") as json_file:
            json_file.write(model_json)
            
        ##Second Layer
        parentmodel1=model
        for layer2units in hidden_neuron_grid:
            model=parentmodel1
            modelname2=modelname+'_'+str(layer2units)
            model=Dense(layer2units, activation=activation)(model)
            model=Dropout(dropout_rate)(model)
            predictions=Dense(y_train.shape[1], activation='sigmoid')(model)
            finalmodel = Model(inputs=inputs, outputs=predictions)
            model_json = finalmodel.to_json()
            print(modelname2)
            with open("models/model_"+modelname2+".json", "w") as json_file:
                json_file.write(model_json)

            ##Third Layer
            parentmodel2=model
            for layer3units in hidden_neuron_grid:
                model=parentmodel2
                modelname3=modelname2+'_'+str(layer3units)
                model=Dense(layer3units, activation=activation)(model)
                model=Dropout(dropout_rate)(model)
                predictions=Dense(y_train.shape[1], activation='sigmoid')(model)
                finalmodel = Model(inputs=inputs, outputs=predictions)
                model_json = finalmodel.to_json()
                print(modelname3)
                with open("models/model_"+modelname3+".json", "w") as json_file:
                    json_file.write(model_json)

                ##Fourth Layer
                parentmodel3=model
                for layer4units in hidden_neuron_grid:
                    model=parentmodel3
                    modelname4=modelname3+'_'+str(layer4units)
                    model=Dense(layer4units, activation=activation)(model)
                    model=Dropout(dropout_rate)(model)
                    predictions=Dense(y_train.shape[1], activation='sigmoid')(model)
                    finalmodel = Model(inputs=inputs, outputs=predictions)
                    model_json = finalmodel.to_json()
                    print(modelname4)
                    with open("models/model_"+modelname4+".json", "w") as json_file:
                        json_file.write(model_json)
