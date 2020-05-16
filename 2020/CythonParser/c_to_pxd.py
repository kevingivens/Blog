import argparse
import os
import sys
import textwrap

# This is not required if you've installed pycparser into
# your site-packages/ with setup.py
# otherwise place this in the pycparser/examples

sys.path.extend(['.', '..'])

from pycparser import parse_file, c_parser, pxd_gen

def translate_to_cython(filename):
    """ Use the Cython Generator module to emit a parsed AST.
    """
    ast = parse_file(filename, use_cpp=True)
    generator = pxd_gen.CythonGen()
    return generator.visit(ast)


def print_to_file(ifile, ofile):
    file_str = f"""
cdef extern from "{os.path.abspath(ifile)}":

{textwrap.indent(translate_to_cython(ifile),'    ')}
"""
    with open(ofile, "w") as f:
        f.write(file_str)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser('Generate pxd from c header files')
    argparser.add_argument('infile', help='name of file to parse')
    argparser.add_argument('outfile', help='name of output file')
    args = argparser.parse_args()

    print_to_file(args.infile, args.outfile)
