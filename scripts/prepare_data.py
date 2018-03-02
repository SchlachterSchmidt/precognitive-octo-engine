"""Prepares the raw dataset.

Creates a sample data subset for prototyping with train, validate and test
sets.

Creates a validation set for the full data set from the training set.
"""


import os
from shutil import copyfile
from glob import glob
import numpy
from numpy.random import permutation


def change_dir(path_name):
    """Util function to safely cd into directory."""
    if os.path.exists(path_name):
        os.chdir(path_name)
        print("pwd: ", os.getcwd())
    else:
        print("unable to cd into %S\naborting...", path_name)
        quit()


FULL_PATH = os.path.realpath(__file__)
PATH, FILENAME = os.path.split(FULL_PATH)
PARENT_DIR = Path(PATH).parent

os.chdir(PARENT_DIR)


DATA_DIR = PARENT_DIR.joinpath('data')

if not os.path.exists(DATA_DIR):
    print("data directory does not exist, download dataset with KAGGLE-CLI\n"
          "aborting..")
    quit()

SAMPLE_TRN_SIZE = 100
SAMPLE_VAL_SIZE = 20
SAMPLE_TEST_SIZE = 1000
VAL_SIZE = 500


DIR_LIST = ['/sample', '/sample/test', '/sample/train', '/sample/train/c0',
            '/sample/train/c1', '/sample/train/c2', '/sample/train/c3',
            '/sample/train/c4', '/sample/train/c5', '/sample/train/c6',
            '/sample/train/c7', '/sample/train/c8', '/sample/train/c9',
            '/sample/valid', 'sample/results', '/sample/valid/c0',
            '/sample/valid/c1', '/sample/valid/c2', '/sample/valid/c3',
            '/sample/valid/c4', '/sample/valid/c5', '/sample/valid/c6',
            '/sample/valid/c7', '/sample/valid/c8', '/sample/valid/c9',
            '/valid', '/valid/c0', '/valid/c1', '/valid/c2', '/valid/c3',
            '/valid/c4', '/valid/c5', '/valid/c6', '/valid/c7', '/valid/c8',
            '/valid/c9', '/results']

for directory in DIR_LIST:
    path = DATA_DIR + directory
    if not os.path.exists(path):
        os.mkdir(DATA_DIR + directory)
        print("created: ", path)
    else:
        print("already exists: ", path)


print('creating /sample/test data set')
change_dir(DATA_DIR + 'test/')
g = glob('*.jpg')
SHUF = numpy.random.permutation(g)
for i in range(SAMPLE_TEST_SIZE):
    copyfile(SHUF[i], DATA_DIR + '/sample/test/' + SHUF[i])


for j in range(10):
    change_dir(DATA_DIR + 'train/c' + str(j))
    g = glob('*.jpg')
    SHUF = numpy.random.permutation(g)

    print('creating /sample/train/c%d data set' % j)
    for i in range(SAMPLE_TRN_SIZE):
        copyfile(SHUF[i], DATA_DIR + '/sample/train/c' + str(j)
                 + '/' + SHUF[i])
    print('creating /sample/valid/c%d data set' % j)
    for i in range(SAMPLE_VAL_SIZE):
        copyfile(SHUF[i], DATA_DIR + '/sample/valid/c' + str(j)
                 + '/' + SHUF[i])
    print('creating /valid/c%d data set' % j)
    for i in range(VAL_SIZE):
        copyfile(SHUF[i], DATA_DIR + '/valid/c' + str(j)
                 + '/' + SHUF[i])
