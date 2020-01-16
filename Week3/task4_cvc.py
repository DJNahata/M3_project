import numpy as np
import pickle
import os

from utils.Kernels import histogram_intersection_kernel
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

FEATURE_PATH = "path"

train_data_path = FEATURE_PATH
test_data_path = FEATURE_PATH
train_data = pickle.load(open(train_data_path,'rb'))
test_data = pickle.load(open(test_data_path,'rb'))

K_FOLDS = 5
param_grid = {'C': [0.001389], 'kernel': ['rbf', histogram_intersection_kernel], 'gamma': ['scale'], 'tol': [1.34590]}

cv = GridSearchCV(SVC(), param_grid=PARAM_GRID, cv=K_FOLDS)
cv.fit(train_data, train_labels)

train_score = cv.score(train_data, train_labels)
test_score  = cv.score(test_data, test_labels)

print("Train accuracy score: {}\nTest accuracy score: {}\nBest params: {}\n".format(train_score, test_score, cv.best_params_))
print("All results: {}".format(cv.cv_results_))
