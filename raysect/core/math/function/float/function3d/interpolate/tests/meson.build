target_path = 'raysect/core/math/function/float/function3d/interpolate/tests'

# source files
py_files = ['test_interpolator_3d.py', '__init__.py']
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

subdir('data')
