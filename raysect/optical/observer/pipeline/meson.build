target_path = 'raysect/optical/observer/pipeline'

# source files
py_files = ['colormaps.py', '__init__.py']
pyx_files = ['rgb.pyx', 'bayer.pyx']
pxd_files = ['bayer.pxd', 'rgb.pxd', '__init__.pxd']
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

subdir('mono')
subdir('spectral')
