from Cython.Compiler.CmdLine import parse_command_line
from Cython.Compiler.Main import create_default_resultobj, CompilationSource
from Cython.Compiler import Pipeline
from Cython.Compiler.Scanning import FileSourceDescriptor
from Cython.Compiler.Visitor import TreeVisitor, PrintTree
from Cython.Compiler.Nodes import *
from Cython.Compiler.ExprNodes import *

import os
import sys

class MyVisitor(TreeVisitor):

    def visit_ModuleNode(self, node):
        print("in %s %s" % (node.child_attrs, self.access_path))
        #print("in %s" % (node.value, self.access_path))
        self.visitchildren(node)

    def visit_CDefExternNode(self, node):
        print("in %s %s" % (node.child_attrs, self.access_path))
        self.visitchildren(node)

    def visit_CStructOrUnionDefNode(self, node):
        print("in %s %s" % (node.child_attrs, self.access_path))
        self.visitchildren(node)

    def visit_StatListNode(self, node):
        print("in %s %s" % (node.child_attrs, self.access_path))
        self.visitchildren(node)

    def visit_CTypeDefNode(self, node):
        print("in %s %s" % (node.child_attrs, self.access_path))
        self.visitchildren(node)

    def visit_CSimpleBaseTypeNode(self, node):
        print("in %s %s" % (node.child_attrs, self.access_path))
        self.visitchildren(node)

    def visit_CPtrDeclaratorNode(self, node):
        print("in %s %s" % (node.child_attrs, self.access_path))
        self.visitchildren(node)

    def visit_CNameDeclaratorNode(self, node):
        print("in %s %s" % (node.child_attrs, self.access_path))
        self.visitchildren(node)

    def visit_CVarDefNode(self, node):
        print("in %s %s" % (node.child_attrs, self.access_path))
        self.visitchildren(node)

    def visit_CFuncDeclaratorNode(self, node):
        print("in %s %s" % (node.child_attrs, self.access_path))
        self.visitchildren(node)

    def visit_CArgDeclNode(self, node):
        print("in %s %s" % (node.child_attrs, self.access_path))
        self.visitchildren(node)

def get_results(root):
    def iter_bodies(tree):
        #try:
        #    for n in tree.body.stats[0].stats:
        #        # cimports at head of file
        #        yield n
        #except:
        #    pass
        if hasattr(tree.body, "stats"):
            for s in tree.body.stats:
                if isinstance(s, CDefExternNode):
                    body = s.body
                    if hasattr(body, "stats"):
                        for node in body.stats:
                            yield node
                    else:
                        yield body
        elif hasattr(tree.body, "body"):
            body = tree.body.body
            yield body
        else:
            raise Exception("parse_pxd_file failed: no valied .pxd file !")

    result = []
    for body in iter_bodies(root):
        if body is not None:
            result.append(body)
        else:
            for node in getattr(body, "stats", []):
                if node is not None:
                    result.append(node)
    return result


def parse_pxd_file(path):
    options, sources = parse_command_line(["", path])

    path = os.path.abspath(path)
    basename = os.path.basename(path)
    name, ext = os.path.splitext(basename)

    source_desc = FileSourceDescriptor(path, basename)
    source = CompilationSource(source_desc, name, os.getcwd())
    result = create_default_resultobj(source, options)
    # print(dir(result))

    context = options.create_context()
    pipeline = Pipeline.create_pyx_pipeline(context, options, result)
    context.setup_errors(options, result)
    root = pipeline[0](source)  # only parser

    #print(root.body.body.stats[3].dump())

    #import pdb; pdb.set_trace()

    return root


if __name__ == "__main__":

    in_file = 'example.pxd'
    #print(parse_pxd_file(in_file))
    root = parse_pxd_file(in_file)

    #MyVisitor().visit(root)

    #PrintTree()(root)
    #result = get_results(root)
    #print(result)
    #for node in iter_bodies(root):
    #    print(node)
    #MyVisitor().visit(root.body)
