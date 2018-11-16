from distutils.core import setup
from Cython.Build import cythonize
import numpy
import os

os.environ["CC"] = "g++"

setup(ext_modules=cythonize('fastdtw_cython.pyx'),include_dirs=[numpy.get_include()])
