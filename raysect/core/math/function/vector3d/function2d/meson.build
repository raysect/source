target_path = 'raysect/core/math/function/vector3d/function2d'

# source files
py_files = ['__init__.py']
pyx_files = ['autowrap.pyx', 'base.pyx', 'utility.pyx', 'blend.pyx', 'constant.pyx']
pxd_files = ['constant.pxd', 'autowrap.pxd', 'utility.pxd', 'blend.pxd', 'base.pxd', '__init__.pxd']
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
