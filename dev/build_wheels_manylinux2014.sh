#!/bin/bash
# To be run from within manylinux2014 docker container.
# Run the command below from the root of the source folder:
# sudo docker run -ti -v $(pwd):/io quay.io/pypa/manylinux2014_x86_64 ./dev/build_wheels_manylinux2014.sh
# Or, to use singularity instead of docker (e.g. on HPC, or where root not available):
# singularity run -B $(pwd):/io -W /tmp -c docker://quay.io/pypa/manylinux2014_x86_64 /io/dev/build_wheels_manylinux2014.sh

set -e
cd /io || exit
VERSION=$(cat raysect/VERSION)
PLAT=manylinux2014_x86_64

# Numpy provides manylinux2010 wheels for Python up to 3.9. So only need manylinux2014
# wheels for 3.10 onwards

# python 3.10
/opt/python/cp310-cp310/bin/python -m build .
auditwheel repair dist/raysect-$VERSION-cp310-cp310-linux_x86_64.whl --plat $PLAT

# python 3.11
/opt/python/cp311-cp311/bin/python -m build .
auditwheel repair dist/raysect-$VERSION-cp311-cp311-linux_x86_64.whl --plat $PLAT

# python 3.12
/opt/python/cp312-cp312/bin/python -m build .
auditwheel repair dist/raysect-$VERSION-cp312-cp312-linux_x86_64.whl --plat $PLAT
