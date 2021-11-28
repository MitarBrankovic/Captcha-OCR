from __future__ import print_function
#import potrebnih biblioteka
import cv2
import collections
import numpy as np
import matplotlib.pylab as plt

# keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
from keras.models import model_from_json

#Sklearn biblioteka sa implementiranim K-means algoritmom
from sklearn import datasets
from sklearn.cluster import KMeans


def create_ann():
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(60, activation='sigmoid'))
    return ann
    
def train_ann(ann, X_train, y_train):

    print("USAO SAM USAO SAM USAO SAM")
    X_train = np.array(X_train, np.float32)
    y_train = np.array(y_train, np.float32)

    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    ann.fit(X_train, y_train, epochs=4200, batch_size=1, verbose = 0, shuffle=False) 
      
    return ann

def serialize_ann(ann):
    model_json = ann.to_json()
    with open("serialization_folder/neuronska.json", "w") as json_file:
        json_file.write(model_json)
    ann.save_weights("serialization_folder/neuronska.h5")
    
def load_trained_ann():
    try:
        json_file = open('serialization_folder/neuronska.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        ann = model_from_json(loaded_model_json)
        ann.load_weights("serialization_folder/neuronska.h5")
        print("Istrenirani model uspesno ucitan.")
        return ann
    except Exception as e:
        return None


def make_alphabet():
    alphabet_first_picture = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                                'N', 'O', 'P', 'Q','R', 'S', 'Š', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ž']

    all_letters = []
    for letter in alphabet_first_picture:
        all_letters.append(letter)

    for letter in alphabet_first_picture:
        letter = letter.lower()
        all_letters.append(letter)

    return all_letters