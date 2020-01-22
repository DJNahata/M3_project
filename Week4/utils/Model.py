from keras.applications import NASNetMobile
from keras.models import Model
from keras.layers import Dense, Activation, GlobalAveragePooling2D, Dropout
from keras.regularizers import l2


class NasNetMob():
    """CLASS::NasNetMob"""
    def __init__(self, img_input=(224,224,3), dropout=None, weight_decay=None):
        model = NASNetMobile(input_shape=img_input, include_top=False, weights='imagenet')
        self.num_new_layers = 2
        x = model.output
        x = GlobalAveragePooling2D()(x)
        if dropout:
            x = Dropout(dropout)(x)
            self.num_new_layers = 3
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
