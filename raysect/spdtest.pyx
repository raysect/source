import time
import numpy as np
from numpy cimport PyArray_SimpleNew, PyArray_FILLWBYTE, NPY_FLOAT64, npy_intp, import_array, ndarray
from cpython cimport array
import array

# required by numpy c-api
import_array()

cdef array.array ARRAY_DOUBLE_TEMPLATE = array.array('d', [])

def run(n=10000, bins=100):

    # array
    start = time.time()
    array_test(n, bins)
    end = time.time()
    elapsed = end - start
    print(f'array: t={elapsed:.3f}s ({elapsed / n * 1e9:.1f}ns per call)')

    # nparray
    start = time.time()
    nparray_py_test(n, bins)
    end = time.time()
    elapsed = end - start
    print(f'array_py: t={elapsed:.3f}s ({elapsed / n * 1e9:.1f}ns per call)')

    # nparray
    start = time.time()
    nparray_c_test(n, bins)
    end = time.time()
    elapsed = end - start
    print(f'nparray_c: t={elapsed:.3f}s ({elapsed / n * 1e9:.1f}ns per call)')


cdef object array_test(int n, int bins):

    cdef array.array d
    cdef double[::1] mv

    for _ in range(n):
        d = array.clone(ARRAY_DOUBLE_TEMPLATE, bins, zero=True)
        mv = d


cdef object nparray_py_test(int n, int bins):

    cdef npy_intp size = bins
    cdef ndarray d
    cdef double[::1] mv

    for _ in range(n):
        d = np.zeros(bins, dtype=np.float64)
        mv = d


cdef object nparray_c_test(int n, int bins):

    cdef npy_intp size = bins
    cdef ndarray d
    cdef double[::1] mv

    for _ in range(n):
        d = PyArray_SimpleNew(1, &size, NPY_FLOAT64)
        PyArray_FILLWBYTE(d, 0)
        mv = d
