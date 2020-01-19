import numpy as np
import pickle
import os

from PIL import Image
from sklearn.feature_extraction import image as skImage
from utils.Kernels import histogram_intersection_kernel, softmax
from utils.Model import Model_MLPatches
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from utils.VisualWords import VisualWords

from sklearn.cluster import MiniBatchKMeans

MODEL_DIR = 'Week3/model9.h5'
DATASET_DIR = 'Databases/MIT_split'
PATCH_SIZE = 64
LAYER = 'dense0'

if __name__ == "__main__":
    print('Loading Model..')
    config_dict = {
        'layers':[{'units':3072,'activation':'relu'},{'units':768,'activation':'relu'}],
        'img_size':PATCH_SIZE,
        'weights_path':MODEL_DIR
    
    }
    mlp_model = Model_MLPatches(config_dict,phase='test',trained=True)
    feat_extractor = mlp_model.get_model_with_layer_name(LAYER)
    features = feat_extractor.output.shape[1]
    print('Loaded')

    train_dir = DATASET_DIR+'/train'
    test_dir = DATASET_DIR+'/test'
    classes = {'coast':0,'forest':1,'highway':2,'inside_city':3,'mountain':4,'Opencountry':5,'street':6,'tallbuilding':7}
    num_patches = int(256/PATCH_SIZE)**2

    if not os.path.isfile('Week3/task5_train_desc.pkl'):
        print('Get train descriptors...')
        num_img = 1881
        count=0
        train_desc = []
        for class_dir in os.listdir(train_dir):
            cls = classes[class_dir]
            for imname in os.listdir(os.path.join(train_dir,class_dir)):
                im = Image.open(os.path.join(train_dir,class_dir,imname))
                patches = skImage.extract_patches_2d(np.array(im), (PATCH_SIZE, PATCH_SIZE), max_patches=num_patches)
                out = feat_extractor.predict(patches/255.)
                train_desc.append(out)
                count += 1
        with open('Week3/task5_train_desc.pkl','wb') as file:
            pickle.dump(train_desc,file)

    if not os.path.isfile('Week3/task5_test_desc.pkl'):
        print('Get test descriptors...')
        num_img = 807
        count=0
        test_desc = []
        for class_dir in os.listdir(test_dir):
            cls = classes[class_dir]
            for imname in os.listdir(os.path.join(test_dir,class_dir)):
                im = Image.open(os.path.join(test_dir,class_dir,imname))
                patches = skImage.extract_patches_2d(np.array(im), (PATCH_SIZE, PATCH_SIZE), max_patches=num_patches)
                out = feat_extractor.predict(patches/255.)
                test_desc.append(out)
                count += 1
        with open('Week3/task5_test_desc.pkl','wb') as file:
            pickle.dump(test_desc,file)

    train_labels = pickle.load(open('Week3/task2_train_label.pkl','rb'))
    test_labels = pickle.load(open('Week3/task2_test_label.pkl','rb'))
    train_desc = pickle.load(open('Week3/task5_train_desc.pkl','rb'))
    test_desc = pickle.load(open('Week3/task5_test_desc.pkl','rb'))

    results_train = []
    results_test = []
    best_params = []
    N_CLUSTERS = [128, 256, 512, 1024]
    for k,n_cluster in enumerate(N_CLUSTERS):
        print('Train Visual Words...')
        visualWords = VisualWords(n_cluster)
        visualWords.fit(train_desc)
        feature_train = visualWords.get_visual_words(train_desc)
        feature_test = visualWords.get_visual_words(test_desc)
        print('Trained.')

        K_FOLDS = 5
        PARAM_GRID = {'C': [0.001389, 0.01, 0.1, 1], 'kernel': [histogram_intersection_kernel], 'gamma': [1e-3, 1e-4, 'scale']}

        cv = GridSearchCV(SVC(), param_grid=PARAM_GRID, cv=K_FOLDS, n_jobs=-1, verbose=5)
        cv.fit(feature_train, train_labels)

        results_train.append(cv.score(feature_train, train_labels))
        results_test.append(cv.score(feature_test, test_labels))
        best_params.append(cv.best_params_)

        print("Train accuracy score: {}\nTest accuracy score: {}\nBest params: {}\n".format(results_train[-1], results_test[-1], cv.best_params_))
        print("All results: {}".format(cv.cv_results_))
        print('CLUSTER: {0}'.format(n_cluster))
    
    with open('Week3/task5_results.pkl','wb') as file:
        pickle.dump((results_train,results_test,best_params),file)
