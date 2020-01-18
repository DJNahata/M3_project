# -- IMPORTS -- #
from utils.Data import PatcherFactory
from utils.Model import Model_MLPatches
import os

# -- CONSTANTS -- #
EPOCHS = 100
PATCH_SIZE  = 64
BATCH_SIZE  = 32
PATCHES_DIR = './Patches_64'
SAVEPATH = './task3_models'

if __name__ == "__main__":
    if not os.path.isdir(SAVEPATH):
        os.mkdir(SAVEPATH)
    # -- DEFINE MODEL COMBINATIONS -- #
    model_comps = [
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
    for ind, config_dict in enumerate(model_comps):
        # -- DEFINE CONSTANTS -- #
        MODEL_NAME = 'model'+str(ind)
        MODEL_FNAME = SAVEPATH+'/'+MODEL_NAME+'.h5'
        BATCH_SIZE = model_comp['batch_size']

        # -- CREATE MODEL -- #
        model = Model_MLPatches(config_dict, phase='train', trained=False)
        model.compile(
            loss='categorical_crossentropy',
            optimizer='sgd',
            metrics=['accuracy']
        )
        print(model.summary())
        plot_model(model, to_file=SAVEPATH+'/'+MODEL_NAME+'_structure.png', show_shapes=True, show_layer_names=True)

        print('Done\n')

        print('Loading Data...')
        train_datagen = ImageDataGenerator(
          rescale=1./255,
          horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            PATCHES_DIR+str(PATCH_SIZE)+'/train',  # this is the target directory
            target_size=(PATCH_SIZE, PATCH_SIZE),  # all images will be resized to PATCH_SIZExPATCH_SIZE
            batch_size=BATCH_SIZE,
            classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
            class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels
  
        validation_generator = test_datagen.flow_from_directory(
            PATCHES_DIR+str(PATCH_SIZE)+'/test',
            target_size=(PATCH_SIZE, PATCH_SIZE),
            batch_size=BATCH_SIZE,
            classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
            class_mode='categorical')
  
        history = model.fit_generator(
            train_generator,
            steps_per_epoch=1881 // BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=validation_generator,
            validation_steps=807 // BATCH_SIZE)

        model.save_weights(MODEL_FNAME)  # always save your weights after training or during training

        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(SAVEPATH+'/'+MODEL_NAME+'_accuracy.png')
        plt.close()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(SAVEPATH+'/'+MODEL_NAME+'_loss.png')
        plt.close()
        """