from setuptools import setup, find_packages, Extension
import sys
import numpy
import os
import os.path as path
import multiprocessing

multiprocessing.set_start_method('fork')

use_cython = True
force = False
profile = False
line_profile = False
annotate = False

if "--skip-cython" in sys.argv:
    use_cython = False
    del sys.argv[sys.argv.index("--skip-cython")]

if "--force" in sys.argv:
    force = True
    del sys.argv[sys.argv.index("--force")]

if "--profile" in sys.argv:
    profile = True
    del sys.argv[sys.argv.index("--profile")]

if "--line-profile" in sys.argv:
    line_profile = True
    del sys.argv[sys.argv.index("--line-profile")]

if "--annotate" in sys.argv:
    annotate = True
    sys.argv.remove("--annotate")

source_paths = ['raysect', 'demos']
compilation_includes = [".", numpy.get_include()]
compilation_args = ['-O3']
cython_directives = {
    # 'auto_pickle': True,
    'language_level': 3
}
setup_path = path.dirname(path.abspath(__file__))

if line_profile:
    compilation_args.append("-DCYTHON_TRACE=1")
    compilation_args.append("-DCYTHON_TRACE_NOGIL=1")
    cython_directives["linetrace"] = True

if use_cython:

    from Cython.Build import cythonize

    # build .pyx extension list
    extensions = []
    for package in source_paths:
        for root, dirs, files in os.walk(path.join(setup_path, package)):
            for file in files:
                if path.splitext(file)[1] == ".pyx":
                    pyx_file = path.relpath(path.join(root, file), setup_path)
                    module = path.splitext(pyx_file)[0].replace("/", ".")
                    extensions.append(Extension(module, [pyx_file], include_dirs=compilation_includes, extra_compile_args=compilation_args),)

    if profile:
        cython_directives["profile"] = True

    # generate .c files from .pyx
    extensions = cythonize(extensions, nthreads=multiprocessing.cpu_count(), force=force, compiler_directives=cython_directives, annotate=annotate)

else:

    # build .c extension list
    extensions = []
    for package in source_paths:
        for root, dirs, files in os.walk(path.join(setup_path, package)):
            for file in files:
                if path.splitext(file)[1] == ".c":
                    c_file = path.relpath(path.join(root, file), setup_path)
                    module = path.splitext(c_file)[0].replace("/", ".")
                    extensions.append(Extension(module, [c_file], include_dirs=compilation_includes, extra_compile_args=compilation_args),)

# parse the package version number
with open(path.join(path.dirname(__file__), 'raysect/VERSION')) as version_file:
    version = version_file.read().strip()

setup(
    name="raysect",
    version=version,
    url="http://www.raysect.org",
    author="Dr Alex Meakins et al.",
    author_email="developers@raysect.org",
    description='A Ray-tracing Framework for Science and Engineering',
    license="BSD",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Cython",
        "Programming Language :: Python :: 3",
        "Topic :: Multimedia :: Graphics :: 3D Rendering",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    install_requires=['numpy', 'matplotlib'],
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    ext_modules=extensions
)
