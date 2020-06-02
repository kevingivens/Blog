# Scipts for generating Cython Bindings from C Source code

The `.pxd` generator is a straightforward adaptation of the C generator from [pycparser](https://github.com/eliben/pycparser).  
It used pycparser for C header parsing.

The `.pyx` generator uses the Cython parser directly.  It's inspired by the [autowrap](https://github.com/uweschmitt/autowrap)] project

Current Usage:
  ```python c_to_pxd.py input.h output.pxd ```

TODO:
-  add setup.py
-  handle enums
