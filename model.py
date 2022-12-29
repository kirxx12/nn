from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from keras import utils, callbacks
import idx2numpy
import numpy as np
from keras import models


def model_1():
    """Хитросложенная модель, немного лучшие результаты, чем у второй, но дольше учится"""
    model = Sequential([
        Convolution2D(filters=32, kernel_size=(3, 3), padding='valid', input_shape=(28, 28, 1), activation='relu'),
        Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(62, activation='softmax'),
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model

def model_2():
    """Самый простой вариант с двумя нейронами"""
    model = Sequential([
        Flatten(),
        Dense(512, activation='relu'),
        Dense(62, activation='softmax')

    ])
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model

def create_model(model_name,
                 path_to_done_NN: str = "NN4(NN3).h5",
                 part_for_samples: int = 1) -> None:
    """Создает НС по заданной модели"""
    x_train = idx2numpy.convert_from_file('emnist/train-images')
    y_train = idx2numpy.convert_from_file('emnist/train-labels')

    x_test = idx2numpy.convert_from_file('emnist/test-images')
    y_test = idx2numpy.convert_from_file('emnist/test-labels')

    x_train = np.reshape(x_train, (x_train.shape[0], 28, 28, 1))
    x_test = np.reshape(x_test, (x_test.shape[0], 28, 28, 1))

    k = 2
    x_train = x_train[:x_train.shape[0] // k]
    y_train = y_train[:y_train.shape[0] // k]
    x_test = x_test[:x_test.shape[0] // k]
    y_test = y_test[:y_test.shape[0] // k]

    x_train = x_train.astype(np.float32)
    x_train /= 255.0
    x_test = x_test.astype(np.float32)
    x_test /= 255.0

    y_train_cat = utils.to_categorical(y_train, 62)
    x_train_cat = utils.to_categorical(y_train, 62)
    y_test_cat = utils.to_categorical(y_test, 62)


    model = model_1()
    learning_rate_reduction = callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                          patience=3,
                                                          verbose=1,
                                                          factor=0.5,
                                                          min_lr=0.00001)
    model.fit(x_train,
              y_train_cat,
              validation_data=(x_test, y_test_cat),
              callbacks=[learning_rate_reduction],
              batch_size=64,
              epochs=30)

    model.save(path_to_done_NN)

def next_train(path_to_done_NN: str,
               path_to_model: str = 'NN4(NN3).h5',
               part_of_dataset: int = 10):
    model = models.load_model(path_to_model)
    x_train = idx2numpy.convert_from_file('emnist/train-images')
    y_train = idx2numpy.convert_from_file('emnist/train-labels')

    x_test = idx2numpy.convert_from_file('emnist/test-images')
    y_test = idx2numpy.convert_from_file('emnist/test-labels')

    x_train = np.reshape(x_train, (x_train.shape[0], 28, 28, 1))
    x_test = np.reshape(x_test, (x_test.shape[0], 28, 28, 1))

    print(len(x_train), len(y_train), len(x_test), len(y_test))

    k = part_of_dataset
    x_train = x_train[300000:700000]
    y_train = y_train[300000:700000]
    x_test = x_test[:]
    y_test = y_test[:]

    x_train = x_train.astype(np.float32)
    x_train /= 255.0
    x_test = x_test.astype(np.float32)
    x_test /= 255.0

    y_train_cat = utils.to_categorical(y_train, 62)
    x_train_cat = utils.to_categorical(y_train, 62)
    y_test_cat = utils.to_categorical(y_test, 62)

    learning_rate_reduction = callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                          patience=3,
                                                          verbose=1,
                                                          factor=0.5,
                                                          min_lr=0.00001)
    model.fit(x_train,
              y_train_cat,
              validation_data=(x_test, y_test_cat),
              callbacks=[learning_rate_reduction],
              batch_size=64,
              epochs=30)

    model.save(path_to_done_NN)


next_train(path_to_done_NN='NN5.h5')