import os
from glob import glob
from random import shuffle

from sklearn.preprocessing import LabelEncoder


class DatasetManager():

    def __init__(self, path):
        self.path = path
        self.label_encoder = LabelEncoder()

    def load_dataset(self,fraction=1):
        self.train_img_paths = glob(os.path.join(self.path, "train", "*", "*.jpg"))
        self.test_img_paths = glob(os.path.join(self.path, "test", "*", "*.jpg"))
        self.train_labels = [os.path.split(os.path.split(img_path)[0])[-1] for img_path in self.train_img_paths]
        self.test_labels = [os.path.split(os.path.split(img_path)[0])[-1] for img_path in self.test_img_paths]

        self.transform_labels()

        return self.train_img_paths, self.train_labels, self.test_img_paths, self.test_labels

    def transform_labels(self):
        self.label_encoder.fit(self.train_labels)
        self.train_labels = self.label_encoder.transform(self.train_labels)
        self.test_labels = self.label_encoder.transform(self.test_labels)

    def inverse_transform_labels(self):
        self.train_labels = self.label_encoder.inverse_transform(self.train_labels)
        self.test_labels = self.label_encoder.inverse_transform(self.test_labels)
