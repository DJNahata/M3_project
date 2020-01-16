import numpy as np
import pickle
import os

from utils.Kernels import histogram_intersection_kernel
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

LAYER_IDX = 1
BASE_PATH = "task4_data"

train_data_path = BASE_PATH + "/task2_train_feat.pkl"
test_data_path = BASE_PATH + "/task2_test_feat.pkl"

train_data = pickle.load(open(train_data_path,'rb'))
feature_train = np.array([item[LAYER_IDX][0] for item in train_data])
test_data = pickle.load(open(test_data_path,'rb'))
feature_test = np.array([item[LAYER_IDX][0] for item in test_data])

train_labels_path = BASE_PATH + "/task2_train_label.pkl"
test_labels_path = BASE_PATH + "/task2_test_label.pkl"
train_labels = pickle.load(open(train_labels_path,'rb'))
test_labels = pickle.load(open(test_labels_path,'rb'))

print(feature_train.shape)
print(len(train_labels))
K_FOLDS = 5
PARAM_GRID = {'C': [0.001389], 'kernel': ['rbf', histogram_intersection_kernel], 'gamma': ['scale'], 'tol': [1.34590]}

cv = GridSearchCV(SVC(), param_grid=PARAM_GRID, cv=K_FOLDS, n_jobs=-1, verbose=5)
cv.fit(feature_train, train_labels)

train_score = cv.score(feature_train, train_labels)
test_score  = cv.score(feature_test, test_labels)

print("Train accuracy score: {}\nTest accuracy score: {}\nBest params: {}\n".format(train_score, test_score, cv.best_params_))
print("All results: {}".format(cv.cv_results_))
