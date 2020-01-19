# -- IMPORTS -- #
from utils.Model import Model_MLP, Model_MLPatches
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import pickle

MODEL_DIR = 'Week3/models/task3'
DATASET_DIR = 'Databases/MIT_split'
PATCH_SIZE = 64
NUM_PATCHES = int(256/PATCH_SIZE)**2

if __name__ == "__main__":
    model_comp = [
        {'layers':[{'units':2048,'activation':'relu'},{'units':1024,'activation':'relu'}],'img_size':PATCH_SIZE,'batch_size':16},
        {'layers':[{'units':3072,'activation':'relu'},{'units':768,'activation':'relu'}],'img_size':PATCH_SIZE,'batch_size':16},
        {'layers':[{'units':2048,'activation':'relu'}],'img_size':PATCH_SIZE,'batch_size':16},
        {'layers':[{'units':3072,'activation':'relu'}],'img_size':PATCH_SIZE,'batch_size':16},
        {'layers':[{'units':2048,'activation':'relu'},{'units':1024,'activation':'relu'}],'img_size':PATCH_SIZE,'batch_size':32},
        {'layers':[{'units':3072,'activation':'relu'},{'units':768,'activation':'relu'}],'img_size':PATCH_SIZE,'batch_size':32},
        {'layers':[{'units':2048,'activation':'relu'}],'img_size':PATCH_SIZE,'batch_size':32},
        {'layers':[{'units':3072,'activation':'relu'}],'img_size':PATCH_SIZE,'batch_size':32},
        {'layers':[{'units':2048,'activation':'relu'},{'units':1024,'activation':'relu'}],'img_size':PATCH_SIZE,'batch_size':64},
        {'layers':[{'units':3072,'activation':'relu'},{'units':768,'activation':'relu'}],'img_size':PATCH_SIZE,'batch_size':64},
        {'layers':[{'units':2048,'activation':'relu'}],'img_size':PATCH_SIZE,'batch_size':64},
        {'layers':[{'units':3072,'activation':'relu'}],'img_size':PATCH_SIZE,'batch_size':64},
    ]

    directory = DATASET_DIR+'/test'
    total   = 807
    results = []
    classes = {'coast':0,'forest':1,'highway':2,'inside_city':3,'mountain':4,'Opencountry':5,'street':6,'tallbuilding':7}
    for k, config_dict in enumerate(model_comp):

        print('Load Model...')
        model = Model_MLP(config_dict,phase='test',trained=True)
        correct = 0.
        count   = 0
        for class_dir in os.listdir(directory):
            cls = classes[class_dir]
            for imname in os.listdir(os.path.join(directory,class_dir)):
                im = Image.open(os.path.join(directory,class_dir,imname))
                patches = image.extract_patches_2d(np.array(im), (PATCH_SIZE, PATCH_SIZE), max_patches=NUM_PATCHES)
                out = model.predict(patches/255.)
                predicted_cls = np.argmax(softmax(np.mean(out,axis=0)))
                if predicted_cls == cls:
                    correct+=1
                count += 1
                print('Evaluated images: '+str(count)+' / '+str(total), end='\r')
        print('Done!\n')
        print('Test Acc. = '+str(correct/total)+'\n')
        results.append(correct/total)
    with open('results_task3.pkl','wb') as file:
        pickle.dump(results)