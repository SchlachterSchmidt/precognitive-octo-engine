"""Quickly check versions of installed tools."""

import platform
print("python: %s" % platform.python_version())

import keras
print("keras: %s" % keras.__version__)

import theano
print("theano: %s" % theano.__version__)

import pygpu
print("pygpu: %s" % pygpu.__version__)

import sklearn
print("scikit-learn: %s" % sklearn.__version__)

import numpy
print("numpy: %s" % numpy.__version__)

import scipy
print("scipy: %s" % scipy.__version__)
