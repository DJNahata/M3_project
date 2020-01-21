from keras.applications import NASNetMobile
from keras.models import Model
from keras.layers import Dense, Activation, GlobalAveragePooling2D

class NasNetMob():
    """CLASS::NasNetMob"""
    def __init__(self,img_input=(224,224,3)):
        model = NASNetMobile(input_shape=img_input, include_top=False, weights='imagenet')
        x = model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(units=8, activation='softmax', name='predictions')(x)
        self.model = Model(inputs=model.inputs, outputs=x)

    def freeze(self):
        for layer in self.model.layers[-2]:
            layer.trainable = False

    def unfreeze(self):
        for layer in self.model.layers[-2]:
            layer.trainable = True