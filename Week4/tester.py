import os
import numpy as np
import matplotlib.pyplot as plt

from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from utils.Data import DataRetrieval
from utils.Model import NasNetMob

# -- CONSTANTS -- #
DATA_DIR = '/home/grupo07/datasets/MIT_400'
WORK_DIR = '/home/grupo07/task3_work_DA_NReg'
NAME = 'experiment1'

# -- CONFIG DICTIONARY -- #
BATCH_SIZE = 16
BATCH_NORM = False
UNITS_2DENSE = None # int to activate
DROPOUT = None # float to enable
WEIGHT_DECAY = None # float to enable
LR = 1E-3
EARLY_PATIENCE = 10
REDUCE_PATIENCE = 10
REDUCE_FACTOR = 0.1
UNFREEZE_LR = 1e-5
DA_AUG = True
TRAIN_SAMPLES = 10*400
VAL_SAMPLES= 10*400
TEST_SAMPLES = 807
RUN_AFTER_FREEZE = False
EPOCHS = 30
OPTIMIZER = 'Adam'
MOMENTUM = 0.9 
NESTEROV = True

if __name__ == "__main__":

    if not os.path.isdir(WORK_DIR):
        os.mkdir(WORK_DIR)

    print('Loading Data...')
    DR = DataRetrieval(DATA_DIR)
    train_generator = DR.get_train_data(augmentation=DA_AUG, batch_size=BATCH_SIZE)
    val_generator = DR.get_validation_data(batch_size=BATCH_SIZE)
    test_generator = DR.get_test_data(batch_size=BATCH_SIZE)
    print('Loaded.')

    print('Loading Model...')
    nasnetmob = NasNetMob(dense=UNITS_2DENSE, batchnorm=BATCH_NORM, dropout=DROPOUT, weight_decay=WEIGHT_DECAY)
    nasnetmob.freeze()
    early = EarlyStopping(monitor='val_loss',patience=EARLY_PATIENCE)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=REDUCE_FACTOR, patience=REDUCE_PATIENCE)
    print('Loaded.')

    print('Training...')
    if OPTIMIZER == 'Adam':
        optimizer = Adam(lr=LR)
    elif OPTIMIZER == 'Adadelta':
        optimizer = Adadelta(lr=LR)
    elif OPTIMIZER == 'RMSprop':
        optimizer = RMSprop(lr=LR)
    else:
        optimizer = SGD(lr=LR, momentum=MOMENTUM, nesterov=NESTEROV)
    
    nasnetmob.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history = nasnetmob.model.fit_generator(
        train_generator,
        steps_per_epoch=(int(TRAIN_SAMPLES // BATCH_SIZE) + 1),
        nb_epoch=EPOCHS,
        validation_data=val_generator,
        validation_steps=(int(VAL_SAMPLES // BATCH_SIZE) + 1),
        callbacks = [early, reduce_lr]
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
        to_write = 'Optimizer: ' + optimizer_name + \
                       '\nLearning rate: ' + str(LR) + \
                       '\nDropout: ' + str(DROPOUT) + \
                       '\nWeight decay: ' + str(WEIGHT_DECAY) + \
                       '\nBatch_size: ' + str(BATCH_SIZE) + \
                       '\nBatchNorm: ' + str(BATCH_NORM) + \
                       '\nTest loss: ' + str(result[0]) + \
                       '\nTest accuracy: ' + str(result[1]) + \
                       '\nEpochs: ' + str(EPOCHS) + \
                       '\n'
        f.write(to_write)

    if RUN_AFTER_FREEZE:
        nasnetmob.unfreeze()
        if OPTIMIZER == 'Adam':
            optimizer = Adam(lr=LR)
        elif OPTIMIZER == 'Adadelta':
            optimizer = Adadelta(lr=LR)
        elif OPTIMIZER == 'RMSprop':
            optimizer = RMSprop(lr=LR)
        else:
            optimizer = SGD(lr=LR, momentum=MOMENTUM, nesterov=NESTEROV)
        nasnetmob.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        history = nasnetmob.model.fit_generator(
            train_generator,
            steps_per_epoch=(int(TRAIN_SAMPLES // BATCH_SIZE) + 1),
            nb_epoch=EPOCHS,
            validation_data=val_generator,
            validation_steps=(int(VAL_SAMPLES // BATCH_SIZE) + 1),
            callbacks = [early, reduce_lr]
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
            to_write = 'Optimizer: ' + optimizer_name + \
                       '\nLearning rate: ' + str(LR) + \
                       '\nDropout: ' + str(DROPOUT) + \
                       '\nWeight decay: ' + str(WEIGHT_DECAY) + \
                       '\nBatch_size: ' + str(BATCH_SIZE) + \
                       '\nBatchNorm: ' + str(BATCH_NORM) + \
                       '\nTest loss: ' + str(result[0]) + \
                       '\nTest accuracy: ' + str(result[1]) + \
                       '\nEpochs: ' + str(EPOCHS) + \
                       '\n'
            f.write(to_write)
