import os
import sys

# import theano.gpuarray
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Flatten
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils.utils import *

current_dir = os.getcwd()
HOME_DIR = current_dir
DATA_DIR = current_dir+'/data/'

# comment out one of the path options to toggle between sample and full set
# path = DATA_HOME_DIR + '/'
path = DATA_DIR + '/sample/'
train_path = path + '/train/'
val_path = path + '/valid/'
test_path = path + '/test/'
results_path = path + '/results/'

# training variables
batch_size = 10
epochs = 5
learning_rate = 1e-5


# getting training and validation data in batches
gen=image.ImageDataGenerator()
batches = get_in_batches(train_path)
val_batches = gen.flow_from_directory(val_path,
                                      target_size=(224,224),
                                      class_mode='categorical',
                                      shuffle=False,
                                      batch_size=batch_size * 2)
test_batches = gen.flow_from_directory(test_path,
                                       target_size=(224,224),
                                       class_mode='categorical',
                                       shuffle=True,
                                       batch_size=batch_size)


# and getting the classes, labels and filenames for each batch
trn_classes = batches.classes
val_classes = val_batches.classes
trn_labels = to_categorical(batches.classes)
val_labels = to_categorical(val_batches.classes)
trn_filenames = batches.filenames
val_filenames = val_batches.filenames


# defining linear model
model = Sequential([
        BatchNormalization(axis=1, input_shape=(3,224,224)),
        Flatten(),
        Dense(10, activation='softmax')
    ])
model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


# fit_generator() consuming the batches provided by the ImageDataGenerator
model.fit_generator(batches,
                    steps_per_epoch=batches.batch_size,
                    validation_data=val_batches,
                    validation_steps=val_batches.batch_size,
                    epochs=epochs)
model.summary()
