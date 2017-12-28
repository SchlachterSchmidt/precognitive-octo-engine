import sys

from keras.models import Sequential
from keras.layers import BatchNormalization, Flatten, Dense
from keras.optimizers import Adam

from pathlib import Path

sys.path.append(str(Path.cwd().parent))
from utils import *

current_dir = Path.cwd()
HOME_DIR = current_dir.parent
DATA_DIR = HOME_DIR.joinpath('data')

# comment out one of the path options to toggle between sample and full set
# path = DATA_HOME_DIR
path = DATA_DIR.joinpath('sample')
train_path = path.joinpath('train')
val_path = path.joinpath('valid')
test_path = path.joinpath('test')
results_path = path.joinpath('tesults')

# training variables
batch_size = 10
epochs = 5
learning_rate = 1e-5


# getting training and validation data in batches
batches = get_in_batches(train_path)
val_batches = get_in_batches(val_path)
test_batches = get_in_batches(test_path)


# and getting the classes, labels and filenames for each batch
trn_classes = batches.classes
val_classes = val_batches.classes
trn_labels = onehot(batches.classes)
val_labels = onehot(val_batches.classes)
trn_filenames = batches.filenames
val_filenames = val_batches.filenames


# defining linear model
model = Sequential([
        BatchNormalization(axis=1, input_shape=(3, 224, 224)),
        Flatten(),
        Dense(10, activation='softmax')
    ])
model.compile(Adam(lr=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


# fit_generator() consuming the images provided by the get_in_batches method
history = model.fit_generator(batches,
                              steps_per_epoch=batches.batch_size,
                              validation_data=val_batches,
                              validation_steps=val_batches.batch_size,
                              epochs=epochs)


plot_acc_and_loss(history)
