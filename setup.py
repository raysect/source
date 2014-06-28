from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import sys
import numpy
import os
import os.path as path

force = False
profile = False

if "--force" in sys.argv:

    force = True
    del sys.argv[sys.argv.index("--force")]

if "--profile" in sys.argv:

    profile = True
    del sys.argv[sys.argv.index("--profile")]

compilation_includes = [".", numpy.get_include()]

setup_path = path.dirname(path.abspath(__file__))

# build extension list
extensions = []
for root, dirs, files in os.walk(setup_path):

    for file in files:

        if path.splitext(file)[1] == ".pyx":

            pyx_file = path.relpath(path.join(root, file), setup_path)
            module = path.splitext(pyx_file)[0].replace("/", ".")

            extensions.append(Extension(module, [pyx_file], include_dirs=compilation_includes),)

if profile:

    directives = {"profile": True}

else:

    directives = {}

setup(ext_modules=cythonize(extensions, force=force, compiler_directives=directives))

