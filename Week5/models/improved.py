from keras import models
from keras import layers
from keras import optimizers


## MODEL A.1
def improvedA(input_shape=(256,256,3), optimizer=optimizers.Adam()):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(inputs)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=8, kernel_size=1, strides=1, padding='valid')(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Activation('softmax')(x)

    model = models.Model(inputs, x)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def improvedB(input_shape=(256,256,3), optimizer=optimizers.Adam()):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(inputs)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=8, kernel_size=1, strides=1, padding='valid')(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Activation('softmax')(x)

    model = models.Model(inputs, x)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def improvedC(input_shape=(256,256,3), optimizer=optimizers.Adam()):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(inputs)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=8, kernel_size=1, strides=1, padding='valid')(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Activation('softmax')(x)

    model = models.Model(inputs, x)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def improvedD(input_shape=(256,256,3), optimizer=optimizers.Adam()):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(inputs)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=8, kernel_size=1, strides=1, padding='valid')(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Activation('softmax')(x)

    model = models.Model(inputs, x)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

## MODEL A.2
def improvedE(input_shape=(256,256,3), optimizer=optimizers.Adam()):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(inputs)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=8, kernel_size=1, strides=1, padding='valid')(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Activation('softmax')(x)

    model = models.Model(inputs, x)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def improvedF(input_shape=(256,256,3), optimizer=optimizers.Adam()):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(inputs)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=8, kernel_size=1, strides=1, padding='valid')(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Activation('softmax')(x)

    model = models.Model(inputs, x)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def improvedG(input_shape=(256,256,3), optimizer=optimizers.Adam()):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(inputs)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=8, kernel_size=1, strides=1, padding='valid')(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Activation('softmax')(x)

    model = models.Model(inputs, x)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def improvedH(input_shape=(256,256,3), optimizer=optimizers.Adam()):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(inputs)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=8, kernel_size=1, strides=1, padding='valid')(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Activation('softmax')(x)

    model = models.Model(inputs, x)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def improvedI(input_shape=(256,256,3), optimizer=optimizers.Adam()):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(inputs)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=8, kernel_size=1, strides=1, padding='valid')(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Activation('softmax')(x)

    model = models.Model(inputs, x)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def improvedJ(input_shape=(256,256,3), optimizer=optimizers.Adam()):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(inputs)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=8, kernel_size=1, strides=1, padding='valid')(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Activation('softmax')(x)

    model = models.Model(inputs, x)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model