import multiprocessing
import os
import sys
from pathlib import Path

import numpy
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext


if os.name == "nt":
    raise OSError("Raysect does not currently support Windows. Please use a Linux or MacOS system.")
else:
    multiprocessing.set_start_method("fork")

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

source_paths = ["raysect", "demos"]
compilation_includes = [".", numpy.get_include()]
compilation_args = ["-O3"]
cython_directives = {
    # 'auto_pickle': True,
    "language_level": 3
}
setup_path = Path(__file__).parent

if line_profile:
    compilation_args.append("-DCYTHON_TRACE=1")
    compilation_args.append("-DCYTHON_TRACE_NOGIL=1")
    cython_directives["linetrace"] = True

if use_cython:
    from Cython.Build import cythonize

    # build .pyx extension list
    extensions = []
    for package in source_paths:
        for pyx in (setup_path / package).glob("**/*.pyx"):
            pyx_path = pyx.relative_to(setup_path)
            module = ".".join(pyx_path.with_suffix("").parts)
            extensions.append(
                Extension(
                    module,
                    [str(pyx_path)],
                    include_dirs=compilation_includes,
                    extra_compile_args=compilation_args,
                ),
            )

    if profile:
        cython_directives["profile"] = True

    # generate .c files from .pyx
    extensions = cythonize(
        extensions,
        nthreads=multiprocessing.cpu_count(),
        force=force,
        compiler_directives=cython_directives,
        annotate=annotate,
    )

else:
    # build .c extension list
    extensions = []
    for package in source_paths:
        for c_file in (setup_path / package).glob("**/*.c"):
            c_file_path = c_file.relative_to(setup_path)
            module = ".".join(c_file_path.with_suffix("").parts)
            extensions.append(
                Extension(
                    module,
                    [str(c_file_path)],
                    include_dirs=compilation_includes,
                    extra_compile_args=compilation_args,
                ),
            )
# parse the package version number
with (Path(__file__).parent / "raysect/VERSION").open("r") as version_file:
    version = version_file.read().strip()


# Use multiple processes by default for building extensions
class build_ext(_build_ext):
    def finalize_options(self):
        super().finalize_options()
        if self.parallel is None:
            nproc = int(os.getenv("RAYSECT_BUILD_JOBS", str(multiprocessing.cpu_count())))
            self.parallel = nproc


setup(
    name="raysect",
    version=version,
    url="http://www.raysect.org",
    author="Dr Alex Meakins et al.",
    author_email="developers@raysect.org",
    description="A Ray-tracing Framework for Science and Engineering",
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
        "Topic :: Scientific/Engineering :: Physics",
    ],
    install_requires=["numpy<2.0", "matplotlib"],
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    ext_modules=extensions,
    cmdclass={"build_ext": build_ext},
)
