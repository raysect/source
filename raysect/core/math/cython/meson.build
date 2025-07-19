target_path = 'raysect/core/math/cython'

# source files
py_files = ['__init__.py']
pyx_files = ['triangle.pyx', 'utility.pyx', 'transform.pyx', 'tetrahedra.pyx']
pxd_files = ['tetrahedra.pxd', 'triangle.pxd', 'utility.pxd', '__init__.pxd', 'transform.pxd']
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

subdir('tests')
subdir('interpolation')
