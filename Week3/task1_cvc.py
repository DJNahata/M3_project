import os
import getpass

from utils import *
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Reshape, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

#user defined variables
EPOCHS      = 100

model_comps = [
    {'layers':[{'units':256,'activation':'relu'}],'img_size':64,'batch_size':16},
    {'layers':[{'units':512,'activation':'relu'}],'img_size':64,'batch_size':16},
    {'layers':[{'units':1024,'activation':'relu'}],'img_size':64,'batch_size':16},
    {'layers':[{'units':2048,'activation':'relu'}],'img_size':64,'batch_size':16},

    {'layers':[{'units':512,'activation':'relu'},{'units':256,'activation':'relu'}],'img_size':64,'batch_size':16},
    {'layers':[{'units':1024,'activation':'relu'},{'units':512,'activation':'relu'}],'img_size':64,'batch_size':16},
    {'layers':[{'units':2048,'activation':'relu'},{'units':1024,'activation':'relu'}],'img_size':64,'batch_size':16},
    {'layers':[{'units':4096,'activation':'relu'},{'units':2048,'activation':'relu'}],'img_size':64,'batch_size':16},

    {'layers':[{'units':512,'activation':'relu'},{'units':256,'activation':'relu'},{'units':128,'activation':'relu'}],'img_size':64,'batch_size':16},
    {'layers':[{'units':1024,'activation':'relu'},{'units':512,'activation':'relu'},{'units':256,'activation':'relu'}],'img_size':64,'batch_size':16},
    {'layers':[{'units':2048,'activation':'relu'},{'units':1024,'activation':'relu'},{'units':512,'activation':'relu'}],'img_size':64,'batch_size':16},
    {'layers':[{'units':4096,'activation':'relu'},{'units':2048,'activation':'relu'},{'units':1024,'activation':'relu'}],'img_size':64,'batch_size':16}
]

DATASET_DIR = '/home/mcv/datasets/MIT_split'
SAVEPATH = '/home/grupo07/work'

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

    print('Start training...\n')

    # this is the dataset configuration we will use for training
    # only rescaling
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True)

    # this is the dataset configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR+'/train',    # this is the target directory
        target_size=(IMG_SIZE, IMG_SIZE),    # all images will be resized to IMG_SIZExIMG_SIZE
        batch_size=BATCH_SIZE,
        classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
        class_mode='categorical')    # since we use binary_crossentropy loss, we need categorical labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        DATASET_DIR+'/test',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
        class_mode='categorical')

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=1881 // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=807 // BATCH_SIZE)

    print('Done!\n')
    print('Saving the model into '+MODEL_FNAME+' \n')
    model.save_weights(MODEL_FNAME)    # always save your weights after training or during training
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
