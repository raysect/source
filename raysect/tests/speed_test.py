from time import time
from math import sqrt as sqrt
from raysect.core.math import Vector as VectorRS
from raysect.tests.speed_test_functions import *

class Vector():

    def __init__(self, v = (1,0,0)):

        try:

            self.x = v[0]
            self.y = v[1]
            self.z = v[2]

        except:

            raise TypeError("Vector can only be initialised with an indexable object, containing numerical values, of length >= 3 items.")

    def __repr__(self):

        return "Vector(["+str(self.x)+", "+str(self.y)+", "+str(self.z)+")]"

    def __add__(self, v):

        return Vector([self.x + v.x, self.y + v.y, self.z + v.z])

    def __radd__(self, v):

        return Vector([self.x + v.x, self.y + v.y, self.z + v.z])

    def __sub__(self, v):

        return Vector([self.x - v.x, self.y - v.y, self.z - v.z])

    def __mul__(self, m):

        return Vector([self.x * m, self.y * m, self.z * m])

    def __rmul__(self, m):

        return Vector([self.x * m, self.y * m, self.z * m])

    def dot(self, v):

        return self.x * v.x + self.y * v.y + self.z * v.z

    def cross(self, v):

        return Vector([self.y * v.z - self.z * v.y,
                      self.z * v.x - self.x * v.z,
                      self.x * v.y - self.y * v.x])

    def normalise(self):

        m = 1 / sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
        return Vector([self.x * m, self.y * m, self.z * m])
 
 
def test1(n, C):

    v1 = C([5.0, 6.0, 7.0])
    v2 = C([-10.0, 55.0, 23.0])

    for i in range(0, n):

        v3 = v1 + v2

    return v3

def test2(n, C):

    v1 = C([5.0, 6.0, 7.0])
    v2 = C([-10.0, 55.0, 23.0])

    for i in range(0, n):

        v3 = v1 - v2

    return v3


def test3(n, C):

    v1 = C([5.0, 6.0, 7.0])
    v2 = C([-10.0, 55.0, 23.0])

    for i in range(0, n):

        r = v1.dot(v2)

    return r


def test4(n, C):

    v1 = C([5.0, 6.0, 7.0])
    v2 = C([-10.0, 55.0, 23.0])

    for i in range(0, n):

        v3 = v1.cross(v2)
        
    return v3


def test5(n, C):

    incident = C([1,-1,0]).normalise()
    normal = C([0,1,0]).normalise()

    for i in range(0, n):

        reflected = incident - 2.0 * normal * (normal.dot(incident))

    return reflected



#---------------------

# the tests

print("Mathematical function speed tests")
print("---------------------------------\n")

n = 10000000

print("Loop of {:,} operations:".format(n))

print("Test 1: addition\n")

t = time()
test1(n, Vector)
pt = time() - t
print(" - python: {:.1f} ms".format(pt*1000))

t = time()
test1(n, VectorRS)
rt = time() - t
print(" - raysect via python: {:.1f} ms".format(rt*1000))

t = time()
ctest1(n)
ct = time() - t
print(" - raysect via cython: {:.1f} ms".format(ct*1000))

t = time()
cotest1(n)
ot = time() - t
print(" - raysect via optimised cython: {:.1f} ms".format(ot*1000))

print("")
print("raysect (python scope) vs python: {:.3F} times faster".format(pt/rt))
print("raysect (cython scope) vs python: {:.3F} times faster".format(pt/ct))
print("raysect (optimised cython) vs python: {:.3F} times faster".format(pt/ot))
print("")

# ---------------------

n = 10000000

print("Loop of {:,} operations:".format(n))

print("Test 2: subtraction\n")

t = time()
test2(n, Vector)
pt = time() - t
print(" - python: {:.1f} ms".format(pt*1000))

t = time()
test2(n, VectorRS)
rt = time() - t
print(" - raysect via python: {:.1f} ms".format(rt*1000))

t = time()
ctest2(n)
ct = time() - t
print(" - raysect via cython: {:.1f} ms".format(ct*1000))

t = time()
cotest2(n)
ot = time() - t
print(" - raysect via optimised cython: {:.1f} ms".format(ot*1000))

print("")
print("raysect (python scope) vs python: {:.3F} times faster".format(pt/rt))
print("raysect (cython scope) vs python: {:.3F} times faster".format(pt/ct))
print("raysect (optimised cython) vs python: {:.3F} times faster".format(pt/ot))
print("")

# ---------------------

n = 10000000

print("Test 3: dot\n")

print("Loop of {:,} operations:".format(n))

t = time()
test3(n, Vector)
pt = time() - t
print(" - python: {:.1f} ms".format(pt*1000))

t = time()
test3(n, VectorRS)
rt = time() - t
print(" - raysect via python: {:.1f} ms".format(rt*1000))

t = time()
ctest3(n)
ct = time() - t
print(" - raysect via cython: {:.1f} ms".format(ct*1000))

t = time()
cotest3(n)
ot = time() - t
print(" - raysect via optimised cython: {:.1f} ms".format(ot*1000))

print("")
print("raysect (python scope) vs python: {:.3F} times faster".format(pt/rt))
print("raysect (cython scope) vs python: {:.3F} times faster".format(pt/ct))
print("raysect (optimised cython) vs python: {:.3F} times faster".format(pt/ot))
print("")

# ---------------------

n = 5000000

print("Test 4: cross\n")

print("Loop of {:,} operations:".format(n))

t = time()
test4(n, Vector)
pt = time() - t
print(" - python: {:.1f} ms".format(pt*1000))

t = time()
test4(n, VectorRS)
rt = time() - t
print(" - raysect via python: {:.1f} ms".format(rt*1000))

t = time()
ctest4(n)
ct = time() - t
print(" - raysect via cython: {:.1f} ms".format(ct*1000))

t = time()
cotest4(n)
ot = time() - t
print(" - raysect via optimised cython: {:.1f} ms".format(ot*1000))

print("")
print("raysect (python scope) vs python: {:.3F} times faster".format(pt/rt))
print("raysect (cython scope) vs python: {:.3F} times faster".format(pt/ct))
print("raysect (optimised cython) vs python: {:.3F} times faster".format(pt/ot))
print("")

#---------------------

n = 2500000

print("Test 5: compound maths (reflection vector)\n")

print("Loop of {:,} operations:".format(n))

t = time()
test5(n, Vector)
pt = time() - t
print(" - python: {:.1f} ms".format(pt*1000))

t = time()
test5(n, VectorRS)
rt = time() - t
print(" - raysect via python: {:.1f} ms".format(rt*1000))

t = time()
ctest5(n)
ct = time() - t
print(" - raysect via cython: {:.1f} ms".format(ct*1000))

t = time()
cotest5a(n)
ota = time() - t
print(" - raysect via optimised cython (high-level): {:.1f} ms".format(ota*1000))

t = time()
cotest5b(n)
otb = time() - t
print(" - raysect via optimised cython (low-level): {:.1f} ms".format(otb*1000))

print("")
print("raysect (python scope) vs python: {:.3F} times faster".format(pt/rt))
print("raysect (cython scope) vs python: {:.3F} times faster".format(pt/ct))
print("raysect (optimised cython, high-level) vs python: {:.3F} times faster".format(pt/ota))
print("raysect (optimised cython, low-level) vs python: {:.3F} times faster".format(pt/otb))
print("")

