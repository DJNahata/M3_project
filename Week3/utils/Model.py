# -- IMPORTS -- #
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Reshape, Activation
from keras import backend as K

# -- MODEL -- #
class Model_MLP():
    """CLASS::Model_MLP"""
    def __init__(self, config_dict, trained=False):
        model = self.get_model_structure(config_dict)
        if trained:
            model.load_weights(config_dict['weights_path'])
        layer_outputs = [layer.output for layer in model.layers]
        self.model = Model(inputs=model.inputs, outputs=layer_outputs)

    def get_model_structure(self,config_dict):
        model = Sequential()
        size = config_dict['img_size']
        model.add(Reshape((size*size*3,),input_shape=(size, size, 3)))
        for layer in config_dict['layers']:
            model.add(Dense(units=layer['units'], activation=layer['activation']))
        model.add(Dense(units=8, activation='softmax'))
        return model

    def predict(self,tensor):
        return self.model.predict(tensor)

class Model_MLPatches():
    """CLASS::Model_MLPatches"""
    def __init__(self,config_dict, phase='train', trained=False):
        self.model = self.get_model_structure(config_dict, phase)
        if trained:
            self.model.load_weights(config_dict['weights_path'])

    def get_model_with_layer_name(self,layer_name):
        return Model(inputs=self.model.inputs,outputs=self.model.get_layer(layer_name).output)
        
    def get_model_structure(self,config_dict, phase):
        model = Sequential()
        size = config_dict['img_size']
        model.add(Reshape((size*size*3,),input_shape=(size, size, 3)))
        for k,layer in enumerate(config_dict['layers']):
            model.add(Dense(units=layer['units'], activation=layer['activation'],name='dense{0}'.format(k)))
        if phase is 'train':
            model.add(Dense(units=8, activation='softmax'))
        else:
            model.add(Dense(units=8, activation='linear'))
        return model

    def predict(self,tensor):
        return self.model.predict(tensor)