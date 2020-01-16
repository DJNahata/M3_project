import os
import getpass

from utils.Data import NormalizedDataGenerator
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Reshape, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

#user defined variables
EPOCHS = 100

model_comps = [
    {'layers':[{'units':2048,'activation':'relu'},{'units':1024,'activation':'relu'}],'img_size':64,'batch_size':16}
]
classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding']

DATASET_DIR = './MIT_split'
SAVEPATH = './task1_models/test_3'

if not os.path.exists(DATASET_DIR):
    print(Color.RED, 'ERROR: dataset directory '+DATASET_DIR+' do not exists!\n')
    quit()

for ind,model_comp in enumerate(model_comps):

    print('\n\nRUNNING MODEL COMP '+str(ind)+'\n\n')

    MODEL_NAME = 'model'+str(ind)
    MODEL_FNAME = SAVEPATH+'/'+MODEL_NAME+'.h5'

    IMG_SIZE = model_comp['img_size']
    BATCH_SIZE = model_comp['batch_size']

    print('Building MLP model...\n')

    #Build the Multi Layer Perceptron model
    model = Sequential()
    model.add(Reshape((IMG_SIZE*IMG_SIZE*3,),input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    for layer in model_comp['layers']:
        model.add(Dense(units=layer['units'], activation=layer['activation']))
    model.add(Dense(units=8, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy'])

    print(model.summary())
    plot_model(model, to_file=SAVEPATH+'/'+MODEL_NAME+'_structure.png', show_shapes=True, show_layer_names=True)

    print('Done!\n')

    if os.path.exists(MODEL_FNAME):
        print('WARNING: model file '+MODEL_FNAME+' exists and will be overwritten!\n')

    print('Loading Data ...\n')
    train_data = NormalizedDataGenerator(DATASET_DIR,classes,IMG_SIZE)
    print('Train Data Loaded.')
    train_generator = train_data.get_data_generator(BATCH_SIZE)
    print('Train Generator Ready')

    test_data = NormalizedDataGenerator(DATASET_DIR,classes,IMG_SIZE,train=False)
    print('Train Data Loaded.')
    test_generator = test_data.get_data_generator(BATCH_SIZE)
    print('Test Generator Ready')

    print('Start training...\n')
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=1881 // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=test_generator,
        validation_steps=807 // BATCH_SIZE)

    print('Done!\n')
    print('Saving the model into '+MODEL_FNAME+' \n')
    model.save(MODEL_FNAME)    # always save your weights after training or during training
    print('Done!\n')

        # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(SAVEPATH+'/'+MODEL_NAME+'_accuracy.png')
    plt.close()
        # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(SAVEPATH+'/'+MODEL_NAME+'_loss.png')
    plt.close()