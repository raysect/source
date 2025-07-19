target_path = 'raysect/primitive'

# source files
py_files = ['__init__.py']
pyx_files = ['sphere.pyx', 'csg.pyx', 'box.pyx', 'cylinder.pyx', 'cone.pyx', 'utility.pyx', 'parabola.pyx']
pxd_files = ['parabola.pxd', 'cone.pxd', 'csg.pxd', 'box.pxd', 'utility.pxd', 'sphere.pxd', '__init__.pxd', 'cylinder.pxd']
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

subdir('lens')
subdir('mesh')
