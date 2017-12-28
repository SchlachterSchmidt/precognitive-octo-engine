"""Module contains a range of Utility functions.

Module is adapted from:
https://github.com/fastai/courses/blob/master/deeplearning1/nbs/utils.py
"""

from keras.preprocessing import image
from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt


def get_in_batches(dirname,
                   gen=image.ImageDataGenerator(),
                   shuffle=True,
                   batch_size=4,
                   class_mode='categorical',
                   target_size=(224, 224)):
    """
    Wrapper method, take path and return image data iterator.

    To be used with Keras model fit_generator method.

    Official Documentation:
    https://keras.io/preprocessing/image/
    """
    return gen.flow_from_directory(dirname,
                                   target_size=target_size,
                                   class_mode=class_mode,
                                   shuffle=shuffle,
                                   batch_size=batch_size)


def onehot(x):
    """Take np array and returns one-hot encoded version."""
    return to_categorical(x)


def plot_acc_and_loss(history):
    """Take Keras history object and plot accuracy and loss."""

    # first grapth plots accuracy of training and validation sets
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # second graph plots loss of training and validation sets
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
