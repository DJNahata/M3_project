import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from utils.CodeTimer import CodeTimer
from utils.DatasetManager import DatasetManager
from descriptors.SIFT import DenseSIFT
from descriptors.visual_words import VisualWords


"""""""""########################
------- DEFINE PARAMETERS -------
##############################"""

# DEFAULT PARAMETERS
default_codebook_size = 256

default_C = 1
default_kernel = "rbf"
default_gamma = "scale"

dataset_path = r"C:\Users\PC\Documents\Roger\Master\M3\Project\Databases\MIT_split"


"""""""""###############################
------- LOAD TRAIN AND TEST SETS -------
#####################################"""

DatasetManager = DatasetManager(dataset_path)
train_img_paths, train_labels, test_img_paths, test_labels = DatasetManager.load_dataset()


"""""""""####################################
------- EXPERIMENT 1: SIFT PARAMETERS -------
##########################################"""

step_sizes = [8,16]
descriptor_sizes_combinations = [[1],[0.5,1],[0.25,0.5,0.75,1]]

DenseSIFT = DenseSIFT()
VisualWords = VisualWords(default_codebook_size)
for step_size in step_sizes:
    for descriptor_sizes_percentages in descriptor_sizes_combinations:
        # Define descriptor sizes
        descriptor_sizes = sorted(list(set([int(i*step_size) for i in descriptor_sizes_percentages])))

        # Obtain training and test data
        with CodeTimer("Obtain training and test data"):
            # Compute DenseSIFT descriptors for train and test sets
            train_descriptors = DenseSIFT.compute(train_img_paths, step_size, descriptor_sizes)
            test_descriptors = DenseSIFT.compute(test_img_paths, step_size, descriptor_sizes)
            # Obtain visual words for train and test sets
            VisualWords.fit(train_descriptors)
            train_data = VisualWords.get_visual_words(train_descriptors)
            test_data = VisualWords.get_visual_words(test_descriptors)

        # Train and test SVM with default parameters
        train_labels = np.vstack(train_labels)
        test_labels = np.vstack(test_labels)

        with CodeTimer("Train SVM"):
            svm = SVC(C=default_C, kernel=default_kernel, gamma=default_gamma)
            svm.fit(train_data,train_labels)

        with CodeTimer("Test SVM"):
            train_score = svm.score(train_data, train_labels)
            test_score = svm.score(test_data, test_labels)

        print("Step size: {}\nDescriptor size percentages: {}\nTrain accuracy score: {}\nTest accuracy score: {}\n".format(step_size, descriptor_size_percentages, train_score, test_score))
