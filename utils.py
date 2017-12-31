"""Module contains a range of Utility functions.

Some module content is adapted from:
https://github.com/fastai/courses/blob/master/deeplearning1/nbs/utils.py
"""

import os
import datetime
from pathlib import Path

import itertools

import matplotlib.pyplot as plt
import numpy as np

from keras.preprocessing import image
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import confusion_matrix


reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.2,
                                  patience=2,
                                  verbose=1,
                                  mode='min',
                                  min_lr=1e-6)


stop_early = EarlyStopping(monitor='val_loss',
                               patience=3,
                               verbose=1,
                               mode='min',
                               min_delta=0
                               )


def get_in_batches(dirname,
                   augment=False,
                   gen=image.ImageDataGenerator(),
                   shuffle=True,
                   batch_size=32,
                   class_mode='categorical',
                   target_size=(224, 224)):
    """
    Wrapper method, take path and return image data iterator.

    To be used with Keras model fit_generator method.
    If augment is set to True, will use the following params:
        - rotation_range = 15
        - height_shift_range = 0.05
        - shear_range = 0.1
        - channel_shift_range = 20
        - width_shift_range = 0.1

    Official Documentation:
    https://keras.io/preprocessing/image/
    """

    if augment:
        gen = image.ImageDataGenerator(rotation_range=15,
                                       height_shift_range=0.05,
                                       shear_range=0.1,
                                       channel_shift_range=20,
                                       width_shift_range=0.1,
                                       vertical_flip=True)

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


def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    """Plot confusion matrix.

    This function is copied from the scikit docs:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
        
        
def load_image(img_path, show=False):
    """Take image path and retun 4d image tensor.

    Format: batch_size, height, width, channels
    """
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor


def save_model(model, filename):
    """Take model and filename as args and save to file."""
    
    dirname = 'savedModels/' + str(datetime.date.today())
    path = Path.cwd().joinpath(dirname)
    
    json_file = filename + '.json'
    h5_file = filename + '.h5'

    if not path.exists():
        os.mkdirs(path)
    
    model_json = model.to_json()
    with open(json_file, "w") as json_file:
        json_file.write(model_json)

    model.save('CNN_two_convs_30122017_1700.h5')