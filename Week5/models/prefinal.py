from keras import models
from keras import layers
from keras import optimizers
from keras import regularizers


def prefinalA(input_shape=(256,256,3), optimizer=optimizers.Adam(), weight_decay=5e-5):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(inputs)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(filters=8, kernel_size=1, strides=1, padding='valid')(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Activation('softmax')(x)

    model = models.Model(inputs, x)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def prefinalB(input_shape=(256,256,3), optimizer=optimizers.Adam(), weight_decay=5e-5):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(inputs)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(filters=8, kernel_size=1, strides=1, padding='valid')(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Activation('softmax')(x)

    model = models.Model(inputs, x)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def prefinalC(input_shape=(256,256,3), optimizer=optimizers.Adam(), weight_decay=5e-5):
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

    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(filters=8, kernel_size=1, strides=1, padding='valid')(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Activation('softmax')(x)

    model = models.Model(inputs, x)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def prefinalD(input_shape=(256,256,3), optimizer=optimizers.Adam(), weight_decay=5e-5):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(inputs)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=8, kernel_size=1, strides=1, padding='valid')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Activation('softmax')(x)

    model = models.Model(inputs, x)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def prefinalE(input_shape=(256,256,3), optimizer=optimizers.Adam(), weight_decay=5e-5):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_regularizer=regularizers.l2(weight_decay))(inputs)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=8, kernel_size=1, strides=1, padding='valid', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Activation('softmax')(x)

    model = models.Model(inputs, x)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def prefinalF(input_shape=(256,256,3), optimizer=optimizers.Adam(), weight_decay=5e-5):
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

    x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(filters=8, kernel_size=1, strides=1, padding='valid')(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Activation('softmax')(x)

    model = models.Model(inputs, x)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def prefinalG(input_shape=(256,256,3), optimizer=optimizers.Adam(), weight_decay=5e-5):
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

    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(filters=8, kernel_size=1, strides=1, padding='valid')(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Activation('softmax')(x)

    model = models.Model(inputs, x)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def prefinalH(input_shape=(256,256,3), optimizer=optimizers.Adam(), weight_decay=5e-5):
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

    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(filters=8, kernel_size=1, strides=1, padding='valid')(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Activation('softmax')(x)

    model = models.Model(inputs, x)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def prefinalI(input_shape=(256,256,3), optimizer=optimizers.Adam(), weight_decay=5e-5):
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

    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(filters=8, kernel_size=1, strides=1, padding='valid')(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Activation('softmax')(x)

    model = models.Model(inputs, x)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def prefinalJ(input_shape=(256,256,3), optimizer=optimizers.Adam(), weight_decay=5e-5):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(inputs)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(filters=8, kernel_size=1, strides=1, padding='valid')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Activation('softmax')(x)

    model = models.Model(inputs, x)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def prefinalK(input_shape=(256,256,3), optimizer=optimizers.Adam(), weight_decay=5e-5):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(inputs)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(filters=8, kernel_size=1, strides=1, padding='valid')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Activation('softmax')(x)

    model = models.Model(inputs, x)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
