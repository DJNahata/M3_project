from glob import glob
from random import shuffle
import shutil
import os

if __name__ == "__main__":
    data_dir = '../Databases/MIT_split'
    save_dir = '../Databases/MIT_400'
    classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding']
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        os.mkdir(save_dir+os.sep+'train')
        os.mkdir(save_dir+os.sep+'test')
        os.mkdir(save_dir+os.sep+'val')


    for category in classes:
        images = glob(data_dir+os.sep+'train'+os.sep+category+os.sep+'*.jpg')
        shuffle(images)
        os.mkdir(save_dir+os.sep+'train'+os.sep+category)
        os.mkdir(save_dir+os.sep+'val'+os.sep+category)
        for img_path in images[:50]:
            shutil.copy(img_path, save_dir+os.sep+'train'+os.sep+category)
        for img_path in images[50:100]:
            shutil.copy(img_path, save_dir+os.sep+'val'+os.sep+category)
    


