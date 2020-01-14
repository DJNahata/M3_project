# -- IMPORTS -- #
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from glob import glob
import numpy as np
import keras
import os

# -- DATA GENERATOR -- #
class DataGenerator():
    """CLASS::DataGenerator:
        >- classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding']
        >- class_mode = 'categorical'"""
    def __init__(self):
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True
        )
        self.test_datagen = ImageDataGenerator(
            rescale=1./255
        )
    
    def get_data_generator(self, data_dir, img_size, batch_size, classes, class_mode):
        train_generator = self.train_datagen.flow_from_directory(
            data_dir+os.sep+'train',  # this is the target directory
            target_size=(img_size, img_size),  # all images will be resized to PATCH_SIZExPATCH_SIZE
            batch_size=batch_size,
            classes = classes,
            class_mode=class_mode  # since we use binary_crossentropy loss, we need categorical labels
        ) 
        test_generator = self.train_datagen.flow_from_directory(
            data_dir+os.sep+'test',  # this is the target directory
            target_size=(img_size, img_size),  # all images will be resized to PATCH_SIZExPATCH_SIZE
            batch_size=batch_size,
            classes = classes,
            class_mode=class_mode  # since we use binary_crossentropy loss, we need categorical labels
        )
        return train_generator,test_generator

# -- NORMALIZATION DATA GENERATOR CLASS -- #
class NormalizedDataGenerator():
    """CLASS::NormalizationDataGenerator:
        >- classes = {
                'coast':0,
                'forest':1,
                'highway':2,
                'inside_city':3,
                'mountain':4,
                'Opencountry':5,
                'street':6,
                'tallbuilding':7
            }
        Normalization is done by mean and std. This is why data needs to be loaded firts."""
    def __init__(self, data_dir, classes, img_size, train=True):
        self.classes = classes
        self.flag = 'train' if train else 'test'
        data_paths = []
        for category in self.classes:
            data_paths.append(glob(data_dir+os.sep+self.flag+os.sep+category+os.sep+'*.jpg'))
        data = []; labels = []
        for k,category in enumerate(data_paths):
            for img in category
                img = load_img(img_path)
                img.resize((img_size,img_size))
                data.append(img_to_array(img))
                labels.append(k)
        self.np_data = np.array(data)
        self.np_labels = np.array(labels)

    def get_data_generator(self, batch_size):
        if self.flag is 'train':
            data_generator = ImageDataGenerator(
                featurewise_center=True,
                featurewise_std_normalization=True,
                horizontal_flip=True
            )
            data_generator.fit(self.np_data)
            print('Mean: {0} -- Std: {1}'.format(data_generator.mean, data_generator.std))
        else:
            data_generator = ImageDataGenerator(
                featurewise_center=True,
                featurewise_std_normalization=True
            )
        return data_generator.flow(self.np_data, self.np_labels, batch_size=batch_size)

# -- Adaptation of the function that appears on utils that makes patches -- #
class PatchDataGenerator():
    """CLASS::PatchDataGenerator:"""
    def __init__(self, in_directory, out_directory):
        if not os.path.exists(out_directory):
            os.makedirs(out_directory)
    
    def get_data_generator(self, patch_size=64):
        total = 2688
        count = 0  
        for split_dir in os.listdir(in_directory):
            if not os.path.exists(os.path.join(out_directory,split_dir)):
                os.makedirs(os.path.join(out_directory,split_dir))
            for class_dir in os.listdir(os.path.join(in_directory,split_dir)):
                if not os.path.exists(os.path.join(out_directory,split_dir,class_dir)):
                    os.makedirs(os.path.join(out_directory,split_dir,class_dir))
                for imname in os.listdir(os.path.join(in_directory,split_dir,class_dir)):
                    count += 1
                    print('Processed images: '+str(count)+' / '+str(total), end='\r')
                    im = Image.open(os.path.join(in_directory,split_dir,class_dir,imname))
                    patches = image.extract_patches_2d(np.array(im), (64, 64), max_patches=1.0)
                    for i,patch in enumerate(patches):
                    patch = Image.fromarray(patch)
                    patch.save(os.path.join(out_directory,split_dir,class_dir,imname.split(',')[0]+'_'+str(i)+'.jpg'))
        print('\n')