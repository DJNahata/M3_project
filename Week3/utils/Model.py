# -- IMPORTS -- #
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Reshape, Activation
from keras import backend as K

# -- MODEL -- #
class MLP_Model():
    """CLASS::MLP_Model"""
        def __init__(self, input_size, phase='train'):
            model = self.get_model_structure(input_size, phase)
            layer_outputs = [layer.output for layer in model.layers]
            self.model = Model(inputs=model.inputs, outputs=layer_outputs)

        def get_model_structure(self,input_size, phase):
            last_act = Activation('softmax') if phase is 'train' else Activation('linear')
            model = Sequential([
                        Reshape((input_size*input_size*3,),input_shape=(input_size, input_size, 3)),
                        Dense(2048),
                        Activation('relu'),
                        Dense(8),
                        last_act
                    ])
            return model