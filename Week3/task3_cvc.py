# -- IMPORTS -- #
from utils.Data import PatcherFactory
from utils.Model import Model_MLPatches

# -- CONSTANTS -- #
EPOCHS = 100
PATCH_SIZE  = 64
BATCH_SIZE  = 16
DATASET_DIR = '/home/mcv/datasets/MIT_split'
PATCHES_DIR = '/home/grupo07/work/data/MIT_split_patches'
SAVEPATH = '/home/grupo07/work'
MODEL_FNAME = '/home/grupo07/work/patch_based_mlp.h5'

if __name__ == "__main__":
    # -- CREATE PATCHES -- #
    patcher_factory = PatcherFactory(DATASET_DIR,PATCHES_DIR+str(PATCH_SIZE))
    patcher_factory.create_patches(PATCH_SIZE)

    # -- DEFINE MODEL COMBINATIONS -- #
    model_comps = [
        {'layers':[{'units':2048,'activation':'relu'},{'units':1024,'activation':'relu'}],'img_size':PATCH_SIZE,'batch_size':16}
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
        """
        Ja ho fet perque es pugui fer load_weights per tant deixe-m'ho així. 
        Si el model després es crea utilitzant tmb la classe, cap problema.
        """

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