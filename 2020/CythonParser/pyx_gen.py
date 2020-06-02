import os
import sys
import textwrap

from Cython.Compiler.CmdLine import parse_command_line
from Cython.Compiler.Main import create_default_resultobj, CompilationSource
from Cython.Compiler import Pipeline
from Cython.Compiler.Scanning import FileSourceDescriptor
from Cython.Compiler.Visitor import TreeVisitor, PrintTree
from Cython.Compiler.Nodes import *
from Cython.Compiler.ExprNodes import *


class PxdVisitor(TreeVisitor):
    """
    Class implementing visitor pattern for traversing PXD parse tree

    visit() visits all nodes a trees and places important nodes in attribute lists
    """

    def __init__(self):
        super(PxdVisitor, self).__init__()
        self.structs = []
        self.funcs = []
        self.type_defs = []
        self.func_args = []

    def visit_ModuleNode(self, node):
        self.mod_name = node.full_module_name
        self.visitchildren(node)

    def visit_CDefExternNode(self, node):
        self.visitchildren(node)

    def visit_CStructOrUnionDefNode(self, node):
        if node.kind == "struct":
            self.structs.append(node.name)
        self.visitchildren(node)

    def visit_StatListNode(self, node):
        self.visitchildren(node)

    def visit_CTypeDefNode(self, node):
        base_type = node.base_type.name
        if isinstance(node.declarator, CPtrDeclaratorNode):
            base_type += "*"

        self.type_defs.append(
          {
            "cname": node.declarator.base.name,
            "type": base_type,
          }
        )
        self.visitchildren(node)

    def visit_CSimpleBaseTypeNode(self, node):
        self.visitchildren(node)

    def visit_CPtrDeclaratorNode(self, node):
        self.visitchildren(node)

    def visit_CNameDeclaratorNode(self, node):
        self.visitchildren(node)

    def visit_CVarDefNode(self, node):
        self.visitchildren(node)

    def visit_CFuncDeclaratorNode(self, node):
        name = node.base.name

        if hasattr(self.access_path[-1][0], 'base_type'):
            return_type = self.access_path[-1][0].base_type.name

        if hasattr(self.access_path[-1][0], 'base'):
            return_type = self.access_path[-2][0].base_type.name + '*'

        self.funcs.append(
            {
              'cname': name,
              'return_type': return_type
            }
        )
        self.visitchildren(node)

    def visit_CArgDeclNode(self, node):

        arg_type = node.base_type.name
        if isinstance(node.declarator, CPtrDeclaratorNode):
            arg_type += "*"

        if hasattr(node, "declarator"):
            if hasattr(node.declarator, "cname"):
                name = node.declarator.name
            if hasattr(node.declarator, "base"):
                name = node.declarator.base.name
        self.func_args.append(
            {
              'func_name': self.access_path[-1][0].base.name,
              'cname': name,
              'type': arg_type,
            }
        )
        self.visitchildren(node)


def parse_pxd_file(path):
    """
    Args
    ====
    path: path to pxd file

    Returns
    =======
    root of parse tree
    """
    options, sources = parse_command_line(["", path])

    path = os.path.abspath(path)
    basename = os.path.basename(path)
    name, ext = os.path.splitext(basename)

    source_desc = FileSourceDescriptor(path, basename)
    source = CompilationSource(source_desc, name, os.getcwd())
    result = create_default_resultobj(source, options)

    context = options.create_context()
    pipeline = Pipeline.create_pyx_pipeline(context, options, result)
    context.setup_errors(options, result)
    root = pipeline[0](source)

    return root


class CastRegionResolver():
    """ Builder class for cast region inside Cython wrapper functions

       These are additional casting steps that can't fit in the Cython func arg

       e.g.
       cdef foo(int bar):
           ***cast region***
           c_foo(<int>bar)
    """

    def __init__(self):
        self.cast_map ={
            "char*" : 'str_cast',
         }

    def str_cast(self, py_name):
        cast_str = f"""py_byte_str = {py_name}.encode('UTF-8')
cdef char* c_{py_name} = py_byte_str"""
        return cast_str

    def resolve_cast_region(self, arg):
        if arg.type in self.cast_map:
            result = getattr(self, self.cast_map[arg.type])(arg.cname)
        else:
            result = None
        return result


class NameResolver():
    """ class for renaming C function to Python class method
        e.g.
        trie_insert() -> Trie.insert()

        attempt to handle special methods
        e.g.
        trie_num_entries() -> Trie.__len__()
        queue_is_empty() -> Queue.__bool__()
    """
    def __init__(self):
        # map python to c types
        self.special_methods = {
            'num_entries': '__len__',
            'is_empty':'__bool__',
        }

    def resolve_name(self, cname):
        """
        """
        py_name = "_".join(cname.split("_")[1:])
        if py_name in self.special_methods:
            return self.special_methods[py_name]
        else:
            return py_name


class ArgResolver():
    """ class mapping c and python types
        used for argument conversion in wrapper functions
    """
    def __init__(self):
        # map python to c types
        self.py_arg_map = {
            'char*': 'str',
            'int': 'int',
            'double': 'double',
        }

        self.c_arg_map = {
            'char*': 'c_',
            'int': '',
            'double': '',
        }

    def resolve_to_c(self, arg):
        c_type = self.c_arg_map.get(arg.type)
        if c_type:
            result = c_type + arg.cname
        else:
            result = '<' + arg.type + '>' + arg.cname
        return result

    def resolve_to_py(self, arg):
        py_type = self.py_arg_map.get(arg.type)
        if py_type:
            result = py_type + ' ' + arg.cname
        else:
            result = arg.type + ' ' + arg.cname
        return result


class PyClass():
    """ template class for building Python classes form C stucts
        Uses string accumulation to build python class implementation
    """
    def __init__(self, kwargs):
        self.mod_name = kwargs['mod_name']
        self.cname = kwargs['cname']
        self.methods = kwargs['methods']
        self.arg_resolver = ArgResolver()
        self.name_resolver = NameResolver()
        self.cast_region_resolver = CastRegionResolver()

    def __str__(self):
        results =  f"struct name: {self.cname}\n"
        for v in self.methods.values():
            results += str(v) + "\n"
        return results

    def build(self):
        """ build python implementation of PyClass """
        template = f"""
cdef class {self.cname}:

    cdef {self.mod_name}.{self.cname}* _this_ptr
"""
        template += self.build_init()
        template += self.build_dealloc()
        for v in self.methods.values():
            template += self.build_method(v)
        return template


    def build_init(self):
        # template components
        new_func = self.methods.pop(f'{self.cname.lower()}_new')
        py_args = self.build_py_args(new_func.args)
        c_args = self.build_c_args(new_func.args)

        # build template
        template = f"""
    def __cinit__(self, {py_args}):
        self._this_ptr = {self.mod_name}.{new_func.cname}({c_args})
        if self._this_ptr is NULL:
            raise MemoryError()
"""
        return template

    def build_dealloc(self):
        # template components
        free_func = self.methods.pop(f'{self.cname.lower()}_free')

        # build template
        template = f"""

    def __dealloc__(self):
        if self._this_ptr is not NULL:
            {self.mod_name}.{free_func.cname}(self._this_ptr)
"""
        return template

    def build_method(self, method):
        """ build methods associated with struct """
        # remove pointer to self (e.g. Trie* trie)
        args = [arg for arg in method.args if (arg.type, arg.cname) != (self.cname+'*', self.cname.lower())]
        # template components
        py_name = self.name_resolver.resolve_name(method.cname)
        py_args = self.build_py_args(args)
        py_cast_region = self.build_py_cast_region(args)
        c_args = self.build_c_args(args)
        py_return_type = self.build_py_return_type(method.return_type)

        # build template
        template = f"""

    cdef {py_name}(self{py_args}):{py_cast_region}
        return {py_return_type}{self.mod_name}.{method.cname}(self._this_ptr{c_args})
"""
        return template

    def build_py_args(self, args):
        """cast c arg list to py arg list if necessary, in order to expose py func
        """
        py_args = [self.arg_resolver.resolve_to_py(arg) for arg in args]
        if py_args:
            result= ", " + textwrap.fill(", ".join(py_args), width=70)
        else:
            result = ""
        return result

    def build_c_args(self, args):
        """cast py arg list back to c args if necessary, in order to call c func
        """
        c_args = [self.arg_resolver.resolve_to_c(arg) for arg in args]
        if c_args:
            result = ", " + textwrap.fill(", ".join(c_args), width=70)
        else:
            result = ""
        return result

    def build_py_cast_region(self, args):
        """py cast region immediately below cython function signature
           for further casting the python args to c args
           e.g.
              cdef foo(bar):
                  ***py cast region***

        """
        py_cast_region = [self.cast_region_resolver.resolve_cast_region(arg) for arg in args]
        # filter out Nones, i.e. args without cast regions
        py_cast_region = [cast for cast in py_cast_region if cast is not None]
        if py_cast_region:
            result = "\n" + textwrap.indent("\n".join(py_cast_region), prefix = ' '*8)
        else:
            result = ""
        return result


    def build_py_return_type(self, return_type):
        """ cast c return type to py return type, if necessary """
        if return_type:
            result = '<' + return_type + '>'
        else:
            result = ''
        return result


class PyFunc():
    def __init__(self, kwargs):
        self.cname = kwargs['cname']
        self.return_type = kwargs['return_type']
        self.args = []

    def __str__(self):
        results = f"func name: {self.cname}\n"
        results += f"func return type: {self.return_type}\n"
        for a in self.args:
            results += str(a)
        return results


class PyFuncArg():
    def __init__(self, kwargs):
        self.cname = kwargs['cname']
        self.type = kwargs['type']

    def __str__(self):
        results = f"arg name: {self.cname}\n"
        results += f"arg type: {self.type}\n"
        return results


def get_struct_name(func_name):
    return func_name.split('_')[0].capitalize()


def group_nodes(visitor):
    """
    group PXD nodes together to form components of Python classes
    structs contain methods which contain func_args
    e.g.

    trie_new()
    trie_free()
    trie_insert()

    -> PyClass(cname=Trie).methods
    """
    structs = {}
    for s in visitor.structs:
        structs[s] = PyClass(
            {
                'mod_name': visitor.mod_name,
                'cname': s,
                'methods': {f['cname']: PyFunc(f) for f in visitor.funcs if s.lower() +"_" in f['cname']},
            }
        )

    for a in visitor.func_args:
        struct_name = get_struct_name(a['func_name'])
        structs[struct_name].methods[a['func_name']].args.append(PyFuncArg(a))
    return structs



if __name__ == "__main__":

    in_file = 'trie.pxd'

    # build root of pxd parse tree
    root = parse_pxd_file(in_file)

    visitor = PxdVisitor()

    # walk tree and store important nodes in attribute lists
    visitor.visit(root)

    #build dict of structs, linking functions associated with each
    structs = group_nodes(visitor)

    for k, v in structs.items():
        print(v.build())
