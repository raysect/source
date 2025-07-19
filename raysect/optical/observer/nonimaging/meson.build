target_path = 'raysect/optical/observer/nonimaging'

# source files
py_files = ['__init__.py']
pyx_files = ['pixel.pyx', 'mesh_camera.pyx', 'fibreoptic.pyx', 'mesh_pixel.pyx', 'sightline.pyx', 'targetted_pixel.pyx']
pxd_files = ['pixel.pxd', 'mesh_camera.pxd', 'sightline.pxd', 'mesh_pixel.pxd', 'fibreoptic.pxd', '__init__.pxd', 'targetted_pixel.pxd']
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

