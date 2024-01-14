from distutils.core import setup
from Cython.Build import cythonize

setup(name="funclib",
      ext_modules=cythonize(
          ["py_func.pyx"], 
          language_level = "3",
      )
)