# to be run from within manylinux2010 docker container
# run the command below from the root of the source folder
# sudo docker run -ti -v $(pwd):/io quay.io/pypa/manylinux2010_x86_64 /bin/bash
# in the container run dev/build_wheels.sh

VERSION=0.8.1
PLAT=manylinux2010_x86_64
CORES=32

cd /io || exit

# python 3.7
/opt/python/cp37-cp37m/bin/python -m pip install cython numpy==1.14.6
/opt/python/cp37-cp37m/bin/python setup.py build_ext -j$CORES
/opt/python/cp37-cp37m/bin/python setup.py bdist_wheel
auditwheel repair dist/raysect-$VERSION-cp37-cp37m-linux_x86_64.whl --plat $PLAT

# python 3.8
/opt/python/cp38-cp38/bin/python -m pip install cython numpy==1.17.5
/opt/python/cp38-cp38/bin/python setup.py build_ext -j$CORES
/opt/python/cp38-cp38/bin/python setup.py bdist_wheel
auditwheel repair dist/raysect-$VERSION-cp38-cp38-linux_x86_64.whl --plat $PLAT

# python 3.9
/opt/python/cp39-cp39/bin/python -m pip install cython numpy==1.19.5
/opt/python/cp39-cp39/bin/python setup.py build_ext -j$CORES
/opt/python/cp39-cp39/bin/python setup.py bdist_wheel
auditwheel repair dist/raysect-$VERSION-cp39-cp39-linux_x86_64.whl --plat $PLAT

# python 3.10
/opt/python/cp310-cp310/bin/python -m pip install cython numpy==1.19.5
/opt/python/cp310-cp310/bin/python setup.py build_ext -j$CORES
/opt/python/cp310-cp310/bin/python setup.py bdist_wheel
auditwheel repair dist/raysect-$VERSION-cp310-cp310-linux_x86_64.whl --plat $PLAT