from keras.applications.nasnet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import os


class DataRetrieval():
    """CLASS:DataRetrieval"""
    def __init__(self, data_dir):
        self.data_dir = data_dir
    
    def get_train_data(self, augmentation=True, img_input=(224,224), batch_size=16):
        if augmentation:
            datagen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                preprocessing_function=preprocess_input, 
                rotation_range=5,
                width_shift_range=0.5,
                height_shift_range=0.5,
                shear_range=0.,
                zoom_range=0.2,
                fill_mode='nearest',
                horizontal_flip=True,
                vertical_flip=False,
                rescale=None
            )
        else:
            datagen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                preprocessing_function=preprocess_input
            )
        return datagen.flow_from_directory(
            self.data_dir+os.sep+'train',
            target_size=img_input,
            batch_size=batch_size,
            class_mode='categorical'
        )

    def get_validation_data(self, img_input=(224,224), batch_size=16):
        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            preprocessing_function=preprocess_input
        )
        return datagen.flow_from_directory(
            self.data_dir+os.sep+'validation',
            target_size=img_input,
            batch_size=batch_size,
            class_mode='categorical'
        )
    
    def get_test_data(self, img_input=(224,224), batch_size=16):
        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            preprocessing_function=preprocess_input
        )
        return datagen.flow_from_directory(
            self.data_dir+os.sep+'test',
            target_size=img_input,
            batch_size=batch_size,
            class_mode='categorical'
        )
