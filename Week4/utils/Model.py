from keras.applications import NASNetMobile
from keras.models import Model
from keras.layers import Dense, Activation, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.regularizers import l2


class NasNetMob():
    """CLASS::NasNetMob"""
    def __init__(self, img_input=(224,224,3), dense=None, batchnorm=False, dropout=None, weight_decay=None):
        model = NASNetMobile(input_shape=img_input, include_top=False, weights='imagenet')
        self.num_new_layers = 2
        x = model.output
        x = GlobalAveragePooling2D()(x)
        if dense:
            x = Dense(units=dense, activation='relu', name='intermidiate')
            self.num_new_layers += 1
        if batchnorm:
            x = BatchNormalization()(x)
            self.num_new_layers += 1
        if dropout and (batchnorm is False):
            x = Dropout(dropout)(x)
            self.num_new_layers += 1
        if weight_decay:
            x = Dense(units=8, activation='softmax', kernel_regularizer=l2(weight_decay), name='predictions')(x)
        else:
            x = Dense(units=8, activation='softmax', name='predictions')(x)
        self.model = Model(inputs=model.inputs, outputs=x)

    def freeze(self):
        for i in range(len(self.model.layers[:-self.num_new_layers])):
            self.model.layers[i].trainable = False

    def unfreeze(self):
        for i in range(len(self.model.layers[:-self.num_new_layers])):
            self.model.layers[i].trainable = True

    def save_weights(self, filepath):
        self.model.save_weights(filepath)
    
    def load_weigths(self, filepath):
        self.model.load_weights(filepath)
