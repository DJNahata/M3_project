import numpy as np
import pickle
import os

from utils.Kernels import histogram_intersection_kernel
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from sklearn.cluster import MiniBatchKMeans


class VisualWords():

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, descriptors):
        self.codebook = MiniBatchKMeans(n_clusters=self.n_clusters,
            verbose=False,
            batch_size=self.n_clusters*20,
            compute_labels=False,
            reassignment_ratio=10**-4,
            random_state=42)
        descriptors = np.vstack(descriptors)
        self.codebook.fit(descriptors)

    def get_visual_words(self, descriptors):
        visual_words = np.empty((len(descriptors), self.n_clusters), dtype=np.float32)
        for i, descriptor in enumerate(descriptors):
            words = self.codebook.predict(descriptor)
            visual_words[i,:] = np.bincount(words, minlength=self.n_clusters)
        return visual_words

BASE_PATH = "task4_data"

train_desc_path = BASE_PATH + "/task3_train_desc.pkl"
test_desc_path = BASE_PATH + "/task3_test_desc.pkl"
train_labels_path = BASE_PATH + "/task2_train_labels.pkl"
test_labels_path = BASE_PATH + "/task2_test_labels.pkl"

train_desc = pickle.load(open(train_desc_path,'rb'))
test_desc = pickle.load(open(test_desc_path,'rb'))

train_labels = pickle.load(open(train_labels_path,'rb'))
test_labels = pickle.load(open(test_labels_path,'rb'))

N_CLUSTERS = 128
visualWords = VisualWords_(N_CLUSTERS)

visualWords.fit(train_desc)
feature_train = visualWords.get_visual_words(train_desc)
feature_test = visualWords.get_visual_words(test_desc)

print(feature_train.shape)
print(len(train_labels))
K_FOLDS = 5
PARAM_GRID = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', histogram_intersection_kernel], 'gamma': [1e-3, 1e-4, 'scale']}

cv = GridSearchCV(SVC(), param_grid=PARAM_GRID, cv=K_FOLDS, n_jobs=-1, verbose=5)
cv.fit(feature_train, train_labels)

train_score = cv.score(feature_train, train_labels)
test_score  = cv.score(feature_test, test_labels)

print("Train accuracy score: {}\nTest accuracy score: {}\nBest params: {}\n".format(train_score, test_score, cv.best_params_))
print("All results: {}".format(cv.cv_results_))
