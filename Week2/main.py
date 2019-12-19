import os
import _pickle as cPickle
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from utils.CodeTimer import CodeTimer
from utils.DatasetManager import DatasetManager
from descriptors.SIFT import DenseSIFT as DenseSIFT_
from descriptors.VisualWords import VisualWords as VisualWords_
from utils.Kernels import histogram_intersection_kernel


class Main():

    def __init__(self, parameters):
        self.default_C = parameters["default_C"]
        self.default_kernel = parameters["default_kernel"]
        self.default_gamma = parameters["default_gamma"]

        self.dataset_path = parameters["dataset_path"]
        self.save_path = parameters["save_path"]

        self.DatasetManager = DatasetManager(self.dataset_path)

    def load_dataset(self):
        self.train_img_paths, self.train_labels, self.test_img_paths, self.test_labels = self.DatasetManager.load_dataset()

    """""""""#################################################
    ------- EXPERIMENT 1: N_CLUSTERS + SIFT PARAMETERS -------
    #######################################################"""

    def experiment1(self):
        n_clusters_ = [128, 256, 384, 512, 640, 768, 896, 1024]

        # Size in memory of training + test set descriptors is 2688*((256/step_size)**2)*128*4*len(descriptor_sizes)
        # Here are some combinations feasible to test considering memory limitations
        step_sizes = [8, 16, 32]
        descriptor_sizes_dict = {8: [[1],[0.5,1]], 16: [[1],[0.5,1],[0.25,0.5,0.75,1]], 32: [[1],[0.5,1],[0.25,0.5,0.75,1]]}

        param_grid = {"C": [self.default_C], "kernel": [self.default_kernel], "gamma": [self.default_gamma]}
        n_folds = 5

        DenseSIFT = DenseSIFT_()
        for n_clusters in n_clusters_:
            VisualWords = VisualWords_(n_clusters)
            for step_size in step_sizes:
                for descriptor_sizes_ in descriptor_sizes_dict[step_size]:
                    # Define descriptor sizes
                    descriptor_sizes = sorted(list(set([int(i*step_size) for i in descriptor_sizes_])))

                    # Obtain training and test data
                    with CodeTimer("Obtain training and test data"):
                        # Check for existing data files already computed
                        train_data_path = os.path.join(self.save_path, "train-test_data", str(n_clusters)+"_"+str(step_size)+"_"+"-".join([str(i) for i in descriptor_sizes])+"_train.dat")
                        test_data_path = train_data_path.replace("_train.dat", "_test.dat")
                        if all(os.path.exists(path) for path in [train_data_path, test_data_path]):
                            train_data = cPickle.load(open(train_data_path, "rb"))
                            test_data = cPickle.load(open(test_data_path, "rb"))
                        else:
                            # Check for DenseSIFT descriptors already computed
                            train_descriptors_path = os.path.join(self.save_path, "train-test_descriptors", str(step_size)+"_"+"-".join([str(i) for i in descriptor_sizes])+"_train.dat")
                            test_descriptors_path = train_descriptors_path.replace("_train.dat", "_test.dat")
                            if all(os.path.exists(path) for path in [train_descriptors_path, test_descriptors_path]):
                                train_descriptors = cPickle.load(open(train_descriptors_path, "rb"))
                                test_descriptors = cPickle.load(open(test_descriptors_path, "rb"))
                            else:
                                # Compute DenseSIFT descriptors for train and test sets
                                train_descriptors = DenseSIFT.compute(self.train_img_paths, step_size, descriptor_sizes)
                                test_descriptors = DenseSIFT.compute(self.test_img_paths, step_size, descriptor_sizes)
                                # Save computed data
                                cPickle.dump(train_descriptors, open(train_descriptors_path, "wb"))
                                cPickle.dump(test_descriptors, open(test_descriptors_path, "wb"))
                            # Obtain visual words for train and test sets
                            VisualWords.fit(train_descriptors)
                            train_data = VisualWords.get_visual_words(train_descriptors)
                            test_data = VisualWords.get_visual_words(test_descriptors)
                            # Save computed data
                            cPickle.dump(train_data, open(train_data_path, "wb"))
                            cPickle.dump(test_data, open(test_data_path, "wb"))

                    # Train SVM with cross-validation 5-fold
                    with CodeTimer("Train SVM"):
                        cv = GridSearchCV(SVC(), param_grid=param_grid, cv=n_folds)
                        cv.fit(train_data, self.train_labels)

                    # Test SVM
                    with CodeTimer("Test SVM"):
                        train_score = cv.score(train_data, self.train_labels)
                        test_score = cv.score(test_data, self.test_labels)

                    print("N_clusters: {}\nStep size: {}\nDescriptor sizes: {}\nTrain accuracy score: {}\nTest accuracy score: {}\n".format(n_clusters, step_size, descriptor_sizes, train_score, test_score))

    """""""""###################################
    ------- EXPERIMENT 2: SVM PARAMETERS -------
    #########################################"""

    def experiment2(self):
        n_clusters = 256
        step_sizes = 16
        descriptor_sizes_ = [0.5, 1]
        param_grid = {"C": np.logspace(-5, 15, 5, base=2), "kernel": ["linear", "rbf", "sigmoid", histogram_intersection_kernel], "gamma": np.logspace(-15, 3, 5, base=2)}
        n_folds = 5

        DenseSIFT = DenseSIFT_()
        VisualWords = VisualWords_(n_clusters)

        # Define descriptor sizes
        descriptor_sizes = sorted(list(set([int(i*step_size) for i in descriptor_sizes_])))

        # Obtain training and test data
        with CodeTimer("Obtain training and test data"):
            # Check for existing data files already computed
            train_data_path = os.path.join(self.save_path, "train-test_data", str(n_clusters)+"_"+str(step_size)+"_"+"-".join([str(i) for i in descriptor_sizes])+"_train.dat")
            test_data_path = train_data_path.replace("_train.dat", "_test.dat")
            if all(os.path.exists(path) for path in [train_data_path, test_data_path]):
                train_data = cPickle.load(open(train_data_path, "rb"))
                test_data = cPickle.load(open(test_data_path, "rb"))
            else:
                # Check for DenseSIFT descriptors already computed
                train_descriptors_path = os.path.join(self.save_path, "train-test_descriptors", str(step_size)+"_"+"-".join([str(i) for i in descriptor_sizes])+"_train.dat")
                test_descriptors_path = train_descriptors_path.replace("_train.dat", "_test.dat")
                if all(os.path.exists(path) for path in [train_descriptors_path, test_descriptors_path]):
                    train_descriptors = cPickle.load(open(train_descriptors_path, "rb"))
                    test_descriptors = cPickle.load(open(test_descriptors_path, "rb"))
                else:
                    # Compute DenseSIFT descriptors for train and test sets
                    train_descriptors = DenseSIFT.compute(train_img_paths, step_size, descriptor_sizes)
                    test_descriptors = DenseSIFT.compute(test_img_paths, step_size, descriptor_sizes)
                    # Save computed data
                    cPickle.dump(train_descriptors, open(train_descriptors_path, "wb"))
                    cPickle.dump(test_descriptors, open(test_descriptors_path, "wb"))
                # Obtain visual words for train and test sets
                VisualWords.fit(train_descriptors)
                train_data = VisualWords.get_visual_words(train_descriptors)
                test_data = VisualWords.get_visual_words(test_descriptors)
                # Save computed data
                cPickle.dump(train_data, open(train_data_path, "wb"))
                cPickle.dump(test_data, open(test_data_path, "wb"))

        # Train SVM with cross-validation 5-fold
        with CodeTimer("Train SVM"):
            cv = GridSearchCV(SVC(), param_grid=param_grid, cv=n_folds)
            cv.fit(train_data, self.train_labels)

        # Test SVM
        with CodeTimer("Test SVM"):
            train_score = cv.score(train_data, self.train_labels)
            test_score = cv.score(test_data, self.test_labels)

        print("Train accuracy score: {}\nTest accuracy score: {}\nBest params: {}\n".format(train_score, test_score, cv.best_params_))
        print("All results: {}".format(cv.cv_results_))


def main():
    parameters = {"default_C": 1,
        "default_kernel": "rbf",
        "default_gamma": "scale",
        "dataset_path": r"C:\Users\PC\Documents\Roger\Master\M3\Project\Databases\MIT_split",
        "save_path": r"C:\Users\PC\Documents\Roger\Master\M3\Project\SavePath"
    }

    M = Main(parameters)
    M.load_dataset()
    M.experiment1()


if __name__ == '__main__':
    main()
