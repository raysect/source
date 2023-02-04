# cython: language_level=3

# Copyright (c) 2014-2023, Dr Alex Meakins, Raysect Project
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#
#     3. Neither the name of the Raysect Project nor the names of its
#        contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# This code is a cython port of the mt19937-64 pseudorandom number generator
# developed by Takuji Nishimura and Makoto Matsumoto. The original license
# follows.

# -----------------------------------------------------------------------------
# A C-program for MT19937-64 (2014/2/23 version).
# Coded by Takuji Nishimura and Makoto Matsumoto.
#
# This is a 64-bit version of Mersenne Twister pseudorandom number
# generator.
#
# Before using, initialize the state by using init_genrand64(seed)
# or init_by_array64(init_key, key_length).
#
# Copyright (C) 2004, 2014, Makoto Matsumoto and Takuji Nishimura,
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#   1. Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#   2. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#   3. The names of its contributors may not be used to endorse or promote
#      products derived from this software without specific prior written
#      permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# References:
# T. Nishimura, ``Tables of 64-bit Mersenne Twisters''
#   ACM Transactions on Modeling and
#   Computer Simulation 10. (2000) 348--357.
# M. Matsumoto and T. Nishimura,
#   ``Mersenne Twister: a 623-dimensionally equidistributed
#     uniform pseudorandom number generator''
#   ACM Transactions on Modeling and
#   Computer Simulation 8. (Jan. 1998) 3--30.
#
# Any feedback is very welcome.
# http://www.math.hiroshima-u.ac.jp/~m-mat/MT/emt.html
# email: m-mat @ math.sci.hiroshima-u.ac.jp (remove spaces)
# -----------------------------------------------------------------------------

from os import urandom as _urandom
from raysect.core.math.vector cimport new_vector3d
from raysect.core.math.point cimport new_point2d, Point3D, new_point3d
from raysect.core.math.cython cimport barycentric_interpolation
from libc.math cimport cos, sin, asin, log, fabs, sqrt, M_PI as PI
from libc.stdint cimport uint64_t, int64_t
cimport cython

DEF NN = 312
DEF MM = 156

# The array for the state vector
cdef uint64_t mt[NN]

# mti == NN+1 means mt[NN] is not initialized
cdef int mti = NN + 1


cdef void init_genrand64(uint64_t seed) nogil:
    """
    Initializes mt[NN] with a seed.
    """

    global mti

    mt[0] = seed
    for mti in range(1, NN):
        mt[mti] = 6364136223846793005UL * (mt[mti - 1] ^ (mt[mti - 1] >> 62)) + mti

    # force word generation
    mti = NN


cdef void init_by_array64(uint64_t init_key[], uint64_t key_length) nogil:
    """
    Initialize with an array.
    :param init_key: The array containing the initializing key.
    :param key_length: The array length.
    """

    cdef:
        unsigned int i, j
        uint64_t k

    init_genrand64(19650218UL)

    i = 1
    j = 0

    for k in range(max(NN, key_length)):

        # non-linear
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 62)) * 3935559000370003845UL)) + init_key[j] + j

        i += 1
        if i >= NN:
            mt[0] = mt[NN - 1]
            i = 1

        j += 1
        if j >= key_length:
            j = 0

    for k in range(NN-1):

        # non-linear
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 62)) * 2862933555777941757UL)) - i
        i += 1
        if i >= NN:
            mt[0] = mt[NN-1]
            i = 1

    mt[0] = 9223372036854775808UL  # 1 << 63, MSB is 1 assuring non-zero initial array


cdef uint64_t _rand_uint64() nogil:
    """
    Generates a random number on [0, 2^64-1] - interval.
    """

    global mti

    cdef:
        int i
        uint64_t x
        uint64_t mag01[2]

    mag01[0] = 0
    mag01[1] = 0xB5026F5AA96619E9UL

    # generate NN words at one time
    if mti >= NN:

        # if init_genrand64() has not been called,
        # a default initial seed is used
        if mti == NN + 1:
            init_genrand64(5489UL)

        # for (i=0;i<NN-MM;i++) {
        for i in range(0, NN - MM):
            x = (mt[i] & 0xFFFFFFFF80000000UL) | (mt[i+1] & 0x7FFFFFFFUL)
            mt[i] = mt[i + MM] ^ (x >> 1) ^ mag01[x & 1]

        for i in range(NN - MM, NN-1):
            x = (mt[i] & 0xFFFFFFFF80000000UL) | (mt[i+1] & 0x7FFFFFFFUL)
            mt[i] = mt[i + (MM - NN)] ^ (x >> 1) ^ mag01[x & 1]

        x = (mt[NN - 1] & 0xFFFFFFFF80000000UL) | (mt[0] & 0x7FFFFFFFUL)
        mt[NN - 1] = mt[MM - 1] ^ (x >> 1) ^ mag01[x & 1]

        mti = 0

    x = mt[mti]
    mti += 1

    x ^= (x >> 29) & 0x5555555555555555UL
    x ^= (x << 17) & 0x71D67FFFEDA60000UL
    x ^= (x << 37) & 0xFFF7EEE000000000UL
    x ^= (x >> 43)

    return x


cpdef seed(object d=None):
    """
    Seeds the random number generator with the specified integer.

    If a seed is not specified the generator is automatically re-seed using the
    system cryptographic random number generator (urandom).

    :param int d: Integer seed.

    .. code-block:: pycon

        >>> from raysect.core.math.random import seed
        >>> seed(1)
    """

    cdef:
        int i
        uint64_t s[NN]

    if d:
        # unpack int into the required number of bytes
        b = d.to_bytes(8*NN, 'big')
    else:
        # if d is not set, source the seed from the system cryptographic random source
        b = _urandom(8*NN)

    for i in range(0, NN):
        s[i] = int.from_bytes(b[i*8:(i+1)*8], byteorder='big')
    init_by_array64(s, NN)


@cython.cdivision(True)
cpdef double uniform():
    """
    Generate random doubles in range [0, 1).

    Values are uniformly distributed.

    :returns: Random double.

    .. code-block:: pycon

        >>> from raysect.core.math.random import uniform
        >>>
        >>> uniform()
        0.7151068954493792
        >>> uniform()
        0.21476630242370853
    """

    return (_rand_uint64() >> 11) * (1.0 / 9007199254740992.0)


# state variables required by the Box-Muller transform
cdef bint _normal_generate = True
cdef double _normal_c1, _normal_c2


cpdef double normal(double mean, double stddev):
    """
    Generates a normally distributed random number.

    The mean and standard deviation of the distribution must be specified.

    :param float mean: The distribution mean.
    :param float stddev: The distribution standard deviation.
    :returns: Random double.

    .. code-block:: pycon

        >>> from raysect.core.math.random import normal
        >>>
        >>> normal(0, 1)
        0.5775399543387388
        >>> normal(0, 1)
        -2.247813575930409
    """

    global _normal_generate, _normal_c1, _normal_c2

    # normals are generated with the Box–Muller transform
    # the transform generates two solutions per evaluation
    _normal_generate = not _normal_generate

    if not _normal_generate:
        return _normal_c1 * sin(_normal_c2) * stddev + mean

    _normal_c1 = sqrt(-2.0 * log(uniform()))
    _normal_c2 = 2.0 * PI * uniform()

    return _normal_c1 * cos(_normal_c2) * stddev + mean


cpdef bint probability(double prob):
    """
    Samples from the Bernoulli distribution where P(True) = prob.

    For example, if probability is 0.8, this function will return True 80% of
    the time and False 20% of the time.

    Values of prob outside the [0, 1] range of probabilities will be clamped to
    the nearest end of the range [0, 1].

    :param double prob: A probability from [0, 1].
    :return: True or False.
    :rtype: bool

    .. code-block:: pycon

        >>> from raysect.core.math.random import probability
        >>>
        >>> probability(0.8)
        True
        >>> probability(0.8)
        True
    """

    return uniform() < prob


cdef Point2D point_disk():
    """
    Returns a random point on a disk of unit radius.

    :rtype: Point2D
    """

    cdef double r = sqrt(uniform())
    cdef double theta = 2.0 * PI * uniform()
    return new_point2d(r * cos(theta), r * sin(theta))


cdef Point2D point_square():
    """
    Returns a random point on a square of unit radius.

    :rtype: Point2D
    """

    return new_point2d(uniform(), uniform())


cdef Point3D point_triangle(Point3D v1, Point3D v2, Point3D v3):

    cdef double temp, alpha, beta, gamma

    # generate barycentric coordinate
    temp = sqrt(uniform())
    alpha = 1 - temp
    beta = uniform() * temp
    gamma = 1 - alpha - beta

    # interpolate vertex coordinates to generate sample point coordinate
    return new_point3d(
        barycentric_interpolation(alpha, beta, gamma, v1.x, v2.x, v3.x),
        barycentric_interpolation(alpha, beta, gamma, v1.y, v2.y, v3.y),
        barycentric_interpolation(alpha, beta, gamma, v1.z, v2.z, v3.z)
    )


cdef Vector3D vector_sphere():
    """
    Generates a random vector on a unit sphere.

    :rtype: Vector3D
    """

    cdef double z = 1.0 - 2.0 * uniform()
    cdef double r = sqrt(max(0, 1.0 - z*z))
    cdef double phi = 2.0 * PI * uniform()
    cdef double x = r * cos(phi)
    cdef double y = r * sin(phi)
    return new_vector3d(x, y, z)


cdef Vector3D vector_hemisphere_uniform():
    """
    Generates a random vector on a unit hemisphere.

    The hemisphere is aligned along the z-axis - the plane that forms the
    hemisphere base lies in the x-y plane.

    :rtype: Vector3D
    """

    cdef double z = uniform()
    cdef double r = sqrt(max(0, 1.0 - z*z))
    cdef double phi = 2.0 * PI * uniform()
    cdef double x = r * cos(phi)
    cdef double y = r * sin(phi)
    return new_vector3d(x, y, z)


cdef Vector3D vector_hemisphere_cosine():
    """
    Generates a cosine-weighted random vector on a unit hemisphere.

    The hemisphere is aligned along the z-axis - the plane that forms the
    hemisphere base lies in the x-y plane.

    :rtype: Vector3D
    """

    cdef double r = sqrt(uniform())
    cdef double phi = 2.0 * PI * uniform()
    cdef double x = r * cos(phi)
    cdef double y = r * sin(phi)
    return new_vector3d(x, y, sqrt(max(0, 1.0 - x*x - y*y)))


cdef Vector3D vector_cone_uniform(double theta):
    """
    Generates a random vector in a cone along the z-axis.

    The angle of the cone is specified with the theta parameter. For speed, no
    checks are performs on the theta parameter, it is up to user to ensure the
    angle is sensible.

    :param float theta: An angle between 0 and 90 degrees.
    :returns: A random Vector3D in the cone defined by theta.
    :rtype: Vector3D
    """

    theta *= 0.017453292519943295 # PI / 180
    cdef double phi = 2.0 * PI * uniform()
    cdef double cos_theta = cos(theta)
    cdef double z = uniform()*(1 - cos_theta) + cos_theta
    cdef double r = sqrt(max(0, 1.0 - z*z))
    cdef double x = r * cos(phi)
    cdef double y = r * sin(phi)
    return new_vector3d(x, y, z)


cdef Vector3D vector_cone_cosine(double theta):
    """
    Generates a cosine-weighted random vector on a cone along the z-axis.

    The angle of the cone is specified with the theta parameter. For speed, no
    checks are performs on the theta parameter, it is up to user to ensure the
    angle is sensible.

    :param float theta: An angle between 0 and 90 degrees.
    :returns: A random Vector3D in the cone defined by theta.
    :rtype: Vector3D
    """

    theta *= 0.017453292519943295 # PI / 180
    cdef double r_max_scaled = asin(theta)
    cdef double r = sqrt(uniform()) * r_max_scaled
    cdef double phi = 2.0 * PI * uniform()
    cdef double x = r * cos(phi)
    cdef double y = r * sin(phi)
    return new_vector3d(x, y, sqrt(max(0, 1.0 - x*x - y*y)))


# initialise random number generator
seed()
