# -- IMPORTS -- #
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Reshape
from keras.preprocessing.image import load_img, img_to_array
from utils.Model import Model_MLP
from utils.Data import DataGetter
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

# -- DEFINE CONSTANTS -- #
config_dict = {
    'layers':[{'units':2048,'activation':'relu'},{'units':1024,'activation':'relu'}],
    'img_size':64,
    'weights_path': './model6.h5'
    }
classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding']
data_dir = '../Databases/MIT_split'
 
if __name__ == "__main__":

    print('Load Model')
    # -- LOAD Model -- #
    model = Model_MLP(config_dict)
    print('Loaded')

    # -- LOAD DATA -- #
    print('Loading train data...')
    train_getter = DataGetter(data_dir, config_dict['img_size'], classes)
    train_getter.load_data()
    print('Loaded.')

    # -- PREDICT DATA -- #
    print('Predicting train data...')
    outputs = []
    labels = []
    for k,(tensor,label) in enumerate(train_getter):
        outputs.append(model.predict(tensor))
        labels.append(label)
        print('Image [{0}] Predicted.'.format(k))
    print('Done')

    with open('./task2_train_feat.pkl','wb') as file:
        pickle.dump(outputs,file)
    with open('./task2_train_label.pkl','wb') as file:
        pickle.dump(labels,file)
    
    # -- LOAD DATA -- #
    print('Loading test data...')
    test_getter = DataGetter(data_dir, config_dict['img_size'], classes, phase='test')
    test_getter.load_data()
    print('Loaded.')

    # -- PREDICT DATA -- #
    print('Predicting test data...')
    outputs = []
    labels = []
    for k,(tensor,label) in enumerate(test_getter):
        outputs.append(model.predict(tensor))
        labels.append(label)
        print('Image [{0}] Predicted.'.format(k))
    print('Done')

    with open('./task2_test_feat.pkl','wb') as file:
        pickle.dump(outputs,file)
    with open('./task2_test_label.pkl','wb') as file:
        pickle.dump(labels,file)