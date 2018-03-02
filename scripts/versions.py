"""Quickly check versions of installed tools."""
import platform
import keras
import numpy
import pygpu
import scipy
import sklearn
import theano


print("python: %s" % platform.python_version())
print("keras: %s" % keras.__version__)
print("theano: %s" % theano.__version__)
print("pygpu: %s" % pygpu.__version__)
print("scikit-learn: %s" % sklearn.__version__)
print("numpy: %s" % numpy.__version__)
print("scipy: %s" % scipy.__version__)
