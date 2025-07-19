target_path = 'raysect/core/math'

# source files
py_files = ['__init__.py']
pyx_files = ['units.pyx', 'normal.pyx', 'point.pyx', 'polygon.pyx', 'quaternion.pyx', 'vector.pyx', '_mat4.pyx', 'random.pyx', '_vec3.pyx', 'transform.pyx', 'statsarray.pyx', 'affinematrix.pyx']
pxd_files = ['units.pxd', 'random.pxd', 'point.pxd', '_vec3.pxd', 'normal.pxd', 'polygon.pxd', 'vector.pxd', 'statsarray.pxd', 'quaternion.pxd', 'affinematrix.pxd', '_mat4.pxd', '__init__.pxd', 'transform.pxd']
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

subdir('function')
subdir('tests')
subdir('cython')
subdir('spatial')
subdir('sampler')
