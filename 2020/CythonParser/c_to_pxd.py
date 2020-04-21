import os
import sys
import textwrap

# This is not required if you've installed pycparser into
# your site-packages/ with setup.py
# otherwise place this in the pycparser/examples

sys.path.extend(['.', '..'])

from pycparser import parse_file, c_parser, cython_gen

def translate_to_cython(filename):
    """ Use the Cython Generator module to emit a parsed AST.
    """
    ast = parse_file(filename, use_cpp=True)
    generator = cython_gen.CythonGen()
    return generator.visit(ast)


def print_to_file(ifile, ofile):
    file_str = f"""
cdef extern from "{os.path.abspath(ifile)}":

{textwrap.indent(translate_to_cython(ifile),'    ')}
"""
    with open(ofile, "w") as f:
        f.write(file_str)


if __name__ == "__main__":

    """ usage: python c_to_pxd.py foo.h foo.pxd

        will generate foo.pxd
        TODO: replace with argparse
    """
    if len(sys.argv) > 1:
        print_to_file(sys.argv[1], sys.argv[2])
    else:
        print("Please provide a filename as argument")
