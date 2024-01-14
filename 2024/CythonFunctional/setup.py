from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

#setup(name="funclib",
#      ext_modules=cythonize(
#          ["py_func.pyx"], 
#          language_level = "3",
#      )
#      extra_compile_arg
#)

from Cython.Distutils import build_ext

setup(
  name = 'Test app',
  ext_modules=[
    Extension('cpp_function_lib',
              sources=['cpp_function_lib.pyx'],
              extra_compile_args=['-std=c++11'],
              language='c++')
    ],
   cmdclass = {'build_ext': build_ext}
)