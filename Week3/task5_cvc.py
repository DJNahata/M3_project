import numpy as np
import pickle
import os

from utils.Kernels import histogram_intersection_kernel
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from utils.VisualWords import VisualWords

from sklearn.cluster import MiniBatchKMeans

BASE_PATH = "task4_data"

train_desc_path = BASE_PATH + "/task3_train_desc.pkl"
test_desc_path = BASE_PATH + "/task3_test_desc.pkl"
train_labels_path = BASE_PATH + "/task3_train_labels.pkl"
test_labels_path = BASE_PATH + "/task3_test_labels.pkl"

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
PARAM_GRID = {'C': [0.001389, 0.01, 0.1, 1, 10], 'kernel': ['rbf', histogram_intersection_kernel], 'gamma': [1e-3, 1e-4, 'scale']}

cv = GridSearchCV(SVC(), param_grid=PARAM_GRID, cv=K_FOLDS, n_jobs=-1, verbose=5)
cv.fit(feature_train, train_labels)

train_score = cv.score(feature_train, train_labels)
test_score  = cv.score(feature_test, test_labels)

print("Train accuracy score: {}\nTest accuracy score: {}\nBest params: {}\n".format(train_score, test_score, cv.best_params_))
print("All results: {}".format(cv.cv_results_))
