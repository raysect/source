project('raysect', 'cython', default_options: ['python.install-env=auto'])

py = import('python').find_installation(pure: false)
numpy = dependency('numpy', method: 'config-tool')
fs = import('fs')

cython_args = ['--annotate']
cython_dependencies = [numpy]

subdir('raysect')
