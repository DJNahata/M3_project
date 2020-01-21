from keras.applications.nasnet import preprocess_input
from keras.applications import NASNetMobile
from keras.utils import plot_model
import os

if __name__ == "__main__":
    print('Loading model....')
    model = NASNetMobile(weights='imagenet')
    print('Loaded')
    plot_model(model, to_file='NASNetMobile.png', show_shapes=True, show_layer_names=True)
    for layer in models.layers:
        print(layer.__class__.__name__)
    print('Loading Without top')
    model = NASNetMobile(weights='imagenet', include_top=False)
    plot_model(model, to_file='NASNetMobile_1.png', show_shapes=True, show_layer_names=True)
