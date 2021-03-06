import os
import numpy as np
import matplotlib.pyplot as plt
import random

from keras.utils import plot_model
from keras import optimizers
from keras import callbacks

from utils.Data import DataRetrieval
from models import *


# Parameters
batch_size = 16
epochs = 200
input_shape = (256,256,3)
learning_rate = 1e-3
data_dir = '/home/mcv/datasets/MIT_split'
work_dir = '/home/grupo07/week5_work/otherC_dropout06_lr_1e-3'
if not os.path.exists(work_dir):
    os.makedirs(work_dir)

# Create model
optimizer = optimizers.Adam(lr=learning_rate)
model = otherC(input_shape=input_shape, optimizer=optimizer, dropout=0.6)

# Print model
model.summary()
plot_model(model, to_file=work_dir+os.sep+'structure.png', show_shapes=True, show_layer_names=True)

# Prepare data
DR = DataRetrieval(data_dir)
train_generator = DR.get_train_data(augmentation=True, batch_size=batch_size)
validation_generator = DR.get_validation_data(batch_size=batch_size)

# Train model
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.2, patience=10)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=20)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=(int(train_generator.n//batch_size)+1),
    nb_epoch=epochs,
    validation_data=validation_generator,
    validation_steps= (int(validation_generator.n//batch_size)+1),
    callbacks=[reduce_lr, early_stopping])

# Summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(work_dir+os.sep+'accuracy.jpg')
plt.close()
# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(work_dir+os.sep+'loss.jpg')
plt.close()
