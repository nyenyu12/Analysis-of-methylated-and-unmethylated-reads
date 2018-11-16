from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extension=[Extension("FDBA_cython",["FDBA_cython.pyx"]
                     #,define_macros=[('CYTHON_TRACE', '1')]
                     ,include_dirs=[numpy.get_include()])]

setup(ext_modules=cythonize(extension))
