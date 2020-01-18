# -- IMPORTS -- #
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from sklearn.feature_extraction import image as skImage
from glob import glob
import numpy as np
import keras
import os

# -- DATA GETTER -- #
class DataGetter():
    """CLASS::DataGetter:"""
    def __init__(self, data_dir, img_size, classes, phase='train'):
        self.phase = phase
        self.data_dir = data_dir
        self.img_size = img_size
        self.classes = classes
        self.data_paths = []
        self.data = []
        self.label = []
    
    def load_data(self):
        for category in self.classes:
            self.data_paths.append(glob(self.data_dir+os.sep+self.phase+os.sep+category+os.sep+'*.jpg'))
        for k, category in enumerate(self.data_paths):
            for img_path in category:
                img = load_img(img_path)
                img = img.resize((self.img_size,self.img_size))
                self.data.append(np.expand_dims(img_to_array(img),axis=0))
                self.label.append(k)
    
    def __getitem__(self, key):
        if not isinstance(key, int):
            raise ValueError('Key has to be of type: int')
        return (self.data[key],self.label[key])
    
    def __len__(self):
        return len(self.data)

# -- PATCHER_FACTORY -- #
class PatcherFactory():
    """CLASS::PatcherFactory"""
    def __init__(self, in_directory, out_directory):
        if not os.path.exists(out_directory):
            os.makedirs(out_directory)
        self.in_dir = in_directory
        self.out_dir = out_directory

    def create_patches(self,patch_size=64):
        total = 2688
        count = 0  
        for split_dir in os.listdir(self.in_dir):
            if not os.path.exists(os.path.join(self.out_dir,split_dir)):
                os.makedirs(os.path.join(self.out_dir,split_dir))

            for class_dir in os.listdir(os.path.join(self.in_dir,split_dir)):
                if not os.path.exists(os.path.join(self.out_dir,split_dir,class_dir)):
                    os.makedirs(os.path.join(self.out_dir,split_dir,class_dir))

                for imname in os.listdir(os.path.join(self.in_dir,split_dir,class_dir)):
                    count += 1
                    im = Image.open(os.path.join(self.in_dir,split_dir,class_dir,imname))
                    patches = skImage.extract_patches_2d(np.array(im), (64, 64), max_patches=4)
                    print('Processed images: '+str(count)+' / '+str(total), end='\r')
                    for i,patch in enumerate(patches):
                        patch = Image.fromarray(patch)
                        patch.save(os.path.join(self.out_dir,split_dir,class_dir,imname.split(',')[0]+'_'+str(i)+'.jpg'))
                    print('\n')

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
            for img_path in category:
                img = load_img(img_path)
                img = img.resize((img_size,img_size))
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
