target_path = 'raysect/core/math/function/vector3d/function2d/tests'

# source files
py_files = ['test_autowrap.py', 'test_float_to_vector3d.py', 'test_constant.py', 'test_base.py', '__init__.py']
pyx_files = []
pxd_files = []
data_files = []

# compile cython
foreach pyx_file: pyx_files
    py.extension_module(
        fs.replace_suffix(pyx_file, ''),
        pyx_file,
        dependencies: cython_dependencies,
        install: true,
        subdir: target_path,
        cython_args: cython_args
    )
endforeach

# add python, pxd and data files to the build
py.install_sources(
    py_files + pxd_files + data_files,
    subdir: target_path
)

