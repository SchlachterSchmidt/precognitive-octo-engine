
# coding: utf-8

# # Final Project DL model Notebook - linear model

# #### imports, settings and constants

# In[1]:


import theano.gpuarray
import os, sys

from keras.models import Sequential
from keras.layers import BatchNormalization, Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from keras import backend
## backend.set_image_dim_ordering('th')

## both lines only needed if importing from utils module
# sys.path.insert(1, os.path.join(sys.path[0], '..'))
# from utils.utils import *

current_dir = os.getcwd()
HOME_DIR = current_dir
DATA_DIR = current_dir+'/data/'

# comment out one of the two path options to toggle between sample directory and all data
# path = DATA_HOME_DIR + '/'
path = DATA_DIR + '/sample/'
train_path = path + '/train/'
val_path = path + '/valid/'
test_path = path + '/test/'
results_path = path + '/results/'

batch_size = 64
epochs = 5


# #### getting training and validation data in batches

# In[2]:


gen=image.ImageDataGenerator()
batches = gen.flow_from_directory(train_path, target_size=(224,224), class_mode='categorical', shuffle=True, batch_size=batch_size)
val_batches = gen.flow_from_directory(val_path, target_size=(224,224), class_mode='categorical', shuffle=False, batch_size=batch_size * 2)
test_batches = gen.flow_from_directory(test_path, target_size=(224,224), class_mode='categorical', shuffle=True, batch_size=batch_size)


# #### and getting the classes, labels and filenames for each batch

# In[3]:


trn_classes = batches.classes
val_classes = val_batches.classes
trn_labels = to_categorical(batches.classes)
val_labels = to_categorical(val_batches.classes)
trn_filenames = batches.filenames
val_filenames = val_batches.filenames


# #### defining linear model

# In[4]:


model = Sequential([
        BatchNormalization(axis=1, input_shape=(3,224,224)),
        Flatten(),
        Dense(10, activation='softmax')
    ])
model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


# #### fit_generator() consuming the batches provided by the ImageDataGenerator to fit the model to the data

# In[5]:


model.fit_generator(batches,
                    steps_per_epoch=batches.samples//batches.batch_size,
                    validation_data=val_batches,
                    validation_steps=val_batches.samples//val_batches.batch_size,
                    epochs=epochs)


# In[6]:


model.summary()

