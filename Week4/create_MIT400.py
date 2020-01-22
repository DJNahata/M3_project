from glob import glob
from random import shuffle
import shutil
import os

if __name__ == "__main__":
    data_dir = '/home/mcv/datasets/MIT_split'
    save_dir = '/home/grupo07/datasets/MIT_400'
    classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding']
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir+os.sep+'train')
        os.makedirs(save_dir+os.sep+'test')
        os.makedirs(save_dir+os.sep+'validation')

    for category in classes:
        images = glob(data_dir+os.sep+'train'+os.sep+category+os.sep+'*.jpg')
        shuffle(images)
        if not os.path.isdir(save_dir+os.sep+'train'+os.sep+category):
            os.makedirs(save_dir+os.sep+'train'+os.sep+category)
        if not os.path.isdir(save_dir+os.sep+'validation'+os.sep+category):
            os.makedirs(save_dir+os.sep+'validation'+os.sep+category)
        for img_path in images[:50]:
            shutil.copy(img_path, save_dir+os.sep+'train'+os.sep+category)
        for img_path in images[50:100]:
            shutil.copy(img_path, save_dir+os.sep+'validation'+os.sep+category)

        images = glob(data_dir+os.sep+'test'+os.sep+category+os.sep+'*.jpg')
        if not os.path.isdir(save_dir+os.sep+'test'+os.sep+category):
            os.makedirs(save_dir+os.sep+'test'+os.sep+category)
        for img_path in images:
            shutil.copy(img_path, save_dir+os.sep+'test'+os.sep+category)

