import os
import numpy as np
import matplotlib.pyplot as plt

from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from utils.Data import DataRetrieval
from utils.Model import NasNetMob

# -- CONSTANTS -- #
DATA_DIR = '/home/grupo07/datasets/MIT_400'
WORK_DIR = '/home/grupo07/work'
NAME = 'experiment'

# -- CONFIG DICTIONARY -- #
BATCH_SIZE = 16
BATCH_NORM = False
UNITS_2DENSE = 512 # int to activate
LR = 1E-3
UNFREEZE_LR = 1e-5
DA_AUG = True
TRAIN_SAMPLES = 400
VAL_SAMPLES= 400
TEST_SAMPLES = 807
RUN_AFTER_FREEZE = False
EPOCHS = 20
OPTIMIZER = SGD
MOMENTUM = 0.9 
NESTEROV = True
DROPOUT = None # float to enable
WEIGHT_DECAY = None # float to enable

if __name__ == "__main__":

    if not os.path.isdir(WORK_DIR):
        os.mkdir(WORK_DIR)

    print('Loading Data...')
    DR = DataRetrieval(data_dir)
    train_generator = DR.get_train_data(augmentation=DA_AUG, batch_size=batch_size)
    val_generator = DR.get_validation_data(batch_size=batch_size)
    test_generator = DR.get_test_data(batch_size=batch_size)
    print('Loaded.')

    print('Loading Model...')
    nasnetmob = NasNetMob(dense=UNITS_2DENSE, batchnorm=BATCH_NORM, dropout=DROPOUT, weight_decay=WEIGHT_DECAY)
    nasnetmob.freeze()
    print('Loaded.')

    print('Training...')
    if OPTIMIZER is SGD:
        optimizer = OPTIMIZER(learning_rate=LR, momentum=MOMENTUM, nesterov=NESTEROV)
    else:
        optimizer = OPTIMIZER(learning_rate=LR)
    
    nasnetmob.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=(int(TRAIN_SAMPLES // BATCH_SIZE) + 1),
        nb_epoch=EPOCHS,
        validation_data=val_generator,
        validation_steps=(int(VAL_SAMPLES // BATCH_SIZE) + 1)
    )
    print('Plotting...')
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(WORK_DIR + os.sep + str(NAME)+'freezed_acc.jpg')
    plt.close()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(WORK_DIR+ os.sep + str(NAME)+'freezed_loss.jpg')
    plt.close()
    print('Ploted.')

    result = nasnetmob.model.evaluate_generator(test_generator, val_samples=TEST_SAMPLES)
    with open(WORK_DIR + os.sep + str(NAME)+'freezed_info.txt', 'w') as f:
        to_write = 'Optimizer: ' + OPTIMIZER.__class__.__name__ + \
                    '\nLearning rate: ' + str(LR) + \
                    '\nBatch_size: ' + str(BATCH_SIZE) + \   
                    '\nDropout: ' + str(DROPOUT) + \
                    '\nBatchNorm: ' + str(BATCH_NORM) + \
                    '\nWeight decay: ' + str(WEIGHT_DECAY) + \
                    '\nTest loss: ' + str(result[0]) + \
                    '\nTest accuracy: ' + str(result[1]) + \
                    '\nEpochs: ' + str(EPOCHS) + \
                    '\n'
        f.write(to_write)

    if RUN_AFTER_FREEZE:
        nasnetmob.unfreeze()
        if OPTIMIZER is SGD:
            optimizer = OPTIMIZER(learning_rate=UNFREEZE_LR, momentum=MOMENTUM, nesterov=NESTEROV)
        else:
            optimizer = OPTIMIZER(learning_rate=UNFREEZE_LR)
        nasnetmob.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        history = model.fit_generator(
            train_generator,
            steps_per_epoch=(int(TRAIN_SAMPLES // BATCH_SIZE) + 1),
            nb_epoch=EPOCHS,
            validation_data=val_generator,
            validation_steps=(int(VAL_SAMPLES // BATCH_SIZE) + 1)
        )
        print('Plotting...')
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(WORK_DIR + os.sep + str(NAME)+'unfreezed_acc.jpg')
        plt.close()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(WORK_DIR+ os.sep + str(NAME)+'unfreezed_loss.jpg')
        plt.close()
        print('Ploted.')

        result = nasnetmob.model.evaluate_generator(test_generator, val_samples=TEST_SAMPLES)
        with open(WORK_DIR + os.sep + str(NAME)+'unfreezed_info.txt', 'w') as f:
            to_write = 'Optimizer: ' + OPTIMIZER.__class__.__name__ + \
                        '\nLearning rate: ' + str(LR) + \
                        '\nBatch_size: ' + str(BATCH_SIZE) + \   
                        '\nDropout: ' + str(DROPOUT) + \
                        '\nBatchNorm: ' + str(BATCH_NORM) + \
                        '\nWeight decay: ' + str(WEIGHT_DECAY) + \
                        '\nTest loss: ' + str(result[0]) + \
                        '\nTest accuracy: ' + str(result[1]) + \
                        '\nEpochs: ' + str(EPOCHS) + \
                        '\n'
            f.write(to_write)
