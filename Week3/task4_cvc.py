import numpy as np
import pickle
import os

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

LAYER_IDX = 1
BASE_PATH = "task4_data"

train_data_path = BASE_PATH + "/task2_train_feat.pkl"
test_data_path = BASE_PATH + "/task2_test_feat.pkl"
#a = pickle.load(open("/Users/joanfontanals/MasterCVC/M3/Project/M3_project/SavePath/data/test128_8_8.dat", 'rb'))
#print(type(a))
#print(a)
#print(a.shape)

train_data = pickle.load(open(train_data_path,'rb'))
print(type(train_data))
print(type(train_data[0]))
print(type(train_data[0][0]))
print(type(train_data[0][0][0]))
print(len(train_data))
print(len(train_data[0]))
print(len(train_data[0][0]))
print(len(train_data[0][0][0]))
#print(train_data)
print(train_data[0])
print(train_data[0][0])
print(np.array(train_data))
test_data = pickle.load(open(test_data_path,'rb'))[LAYER_IDX]

train_labels_path = BASE_PATH + "/task2_train_label.pkl"
test_labels_path = BASE_PATH + "/task2_test_label.pkl"
train_labels = pickle.load(open(train_labels_path,'rb'))
test_labels = pickle.load(open(test_labels_path,'rb'))

print(train_data.shape)
print(test_data.shape)

print(train_labels.shape)
print(test_labels.shape)

def histogram_intersection_kernel(a, b):
    K = np.empty(shape=(a.shape[0], b.shape[0]), dtype=np.float32)
    for i in range(a.shape[0]):
        K[i] = np.sum(np.minimum(a[i], b), axis=1)
    return K

K_FOLDS = 5
PARAM_GRID = {'C': [0.001389], 'kernel': ['rbf', histogram_intersection_kernel], 'gamma': ['scale'], 'tol': [1.34590]}

cv = GridSearchCV(SVC(), param_grid=PARAM_GRID, cv=K_FOLDS)
cv.fit(train_data, train_labels)

train_score = cv.score(train_data, train_labels)
test_score  = cv.score(test_data, test_labels)

print("Train accuracy score: {}\nTest accuracy score: {}\nBest params: {}\n".format(train_score, test_score, cv.best_params_))
print("All results: {}".format(cv.cv_results_))
