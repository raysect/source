#!/bin/bash
# To be run from within manylinux2010 docker container.
# Run the command below from the root of the source folder:
# sudo docker run -ti -v $(pwd):/io quay.io/pypa/manylinux2010_x86_64 ./dev/build_wheels_manylinux2010.sh
# Or, to use singularity instead of docker (e.g. on HPC, or where root not available):
# singularity run -B $(pwd):/io -W /tmp -c docker://quay.io/pypa/manylinux2010_x86_64 /io/dev/build_wheels_manylinux2010.sh

set -e
cd /io || exit
VERSION=$(cat raysect/VERSION)
PLAT=manylinux2010_x86_64

# Numpy provides manylinux2010 wheels only for Python up to 3.9.

# python 3.7
/opt/python/cp37-cp37m/bin/python -m build .
auditwheel repair dist/raysect-$VERSION-cp37-cp37m-linux_x86_64.whl --plat $PLAT

# python 3.8
/opt/python/cp38-cp38/bin/python -m build .
auditwheel repair dist/raysect-$VERSION-cp38-cp38-linux_x86_64.whl --plat $PLAT

# python 3.9
/opt/python/cp39-cp39/bin/python -m build .
auditwheel repair dist/raysect-$VERSION-cp39-cp39-linux_x86_64.whl --plat $PLAT
