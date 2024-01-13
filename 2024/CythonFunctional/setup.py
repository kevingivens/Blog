from distutils.core import setup
from Cython.Build import cythonize

setup(name="func",
      ext_modules=cythonize(["func.pyx"]))