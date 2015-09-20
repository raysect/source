# cython: language_level=3

# Copyright (c) 2015, Dr Alex Meakins, Raysect Project
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

from raysect.core.math.vector cimport new_vector
from raysect.core.math.point cimport new_point
from libc.math cimport cos, sin, sqrt, M_PI as PI
from libc.stdint cimport uint64_t, int64_t
cimport cython

DEF NN = 312
DEF MM = 156

# The array for the state vector
cdef uint64_t mt[NN]

# mti == NN+1 means mt[NN] is not initialized
cdef int mti = NN + 1

# initializes mt[NN] with a seed
cdef void init_genrand64(uint64_t seed):

    global mti

    mt[0] = seed
    for mti in range(1, NN):
        mt[mti] = 6364136223846793005UL * (mt[mti - 1] ^ (mt[mti - 1] >> 62)) + mti

    # force word generation
    mti = NN

# initialize by an array with array-length
# init_key is the array for initializing keys
# key_length is its length
cdef void init_by_array64(uint64_t init_key[], uint64_t key_length):

    cdef:
        unsigned int i, j
        uint64_t k

    init_genrand64(19650218UL)

    i = 1
    j = 0
    if NN > key_length:
        k = NN
    else:
        k = key_length

    # for (; k; k--)
    # TODO: refactor as k is not used! k_max = max(NN, key_length)
    for k in range(k, 0, -1):

        # non-linear
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 62)) * 3935559000370003845UL)) + init_key[j] + j

        i += 1
        if i >= NN:
            mt[0] = mt[NN - 1]
            i = 1

        j += 1
        if j >= key_length:
            j = 0

    #for (k=NN-1; k; k--) {
    # TODO: refactor as k is not used
    for k in range(NN-1, 0, -1):

        # non-linear
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 62)) * 2862933555777941757UL)) - i
        i += 1
        if i >= NN:
            mt[0] = mt[NN-1]
            i = 1

    mt[0] = 9223372036854775808UL  # 1 << 63, MSB is 1 assuring non-zero initial array


# generates a random number on [0, 2^64-1]-interval
cdef uint64_t genrand64_int64():

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


# /* generates a random number on [0, 2^63-1]-interval */
# int64_t genrand64_int63(void)
# {
#     return (int64_t)(genrand64_int64() >> 1);
# }
#
# /* generates a random number on [0,1]-real-interval */
# double genrand64_real1(void)
# {
#     return (genrand64_int64() >> 11) * (1.0/9007199254740991.0);
# }
#
# /* generates a random number on [0,1)-real-interval */
# double genrand64_real2(void)
# {
#     return (genrand64_int64() >> 11) * (1.0/9007199254740992.0);
# }
#
# /* generates a random number on (0,1)-real-interval */
# double genrand64_real3(void)
# {
#     return ((genrand64_int64() >> 12) + 0.5) * (1.0/4503599627370496.0);
# }




def test_random(n):

    cdef:
        int i
        uint64_t init[4]
        uint64_t length

    init[0] = 0x12345UL
    init[1] = 0x23456UL
    init[2] = 0x34567UL
    init[3] = 0x45678UL
    length = 4

    init_by_array64(init, length);

    print("1000 outputs of genrand64_int64()")
    for i in range(n):
        print("{} ".format(genrand64_int64()), end="")
        if i % 5 == 4:
            print()

    # printf("\n1000 outputs of genrand64_real2()\n");
    # for (i=0; i<1000; i++) {
    #   printf("%10.8f ", genrand64_real2());
    #   if (i%5==4) printf("\n");
    # }








cpdef double random():
    """
    Generate random doubles in range [0, 1).

    Values are uniformly distributed.

    :return: Random double.
    """
    return 1 #genrand64_real2()


cpdef bint probability(double prob):
    """
    Samples from the Bernoulli distribution where P(True) = prob.

    For example, if probability is 0.8, this function will return True 80% of
    the time and False 20% of the time.

    Values of prob outside the [0, 1] range of probabilities will be clamped to
    the nearest end of the range [0, 1].

    :param double prob: A probability from [0, 1].
    :return: True or False.
    """

    return 1 #genrand64_real2() < prob


cpdef Point point_disk():
    """
    Returns a random point on a disk of unit radius.

    The disk lies in the x-y plane and is centered at the origin.

    :return: A Point on the disk.
    """

    cdef double r = sqrt(random())
    cdef double theta = 2.0 * PI * random()
    return new_point(r * cos(theta), r * sin(theta), 0)


# cpdef Vector vector_sphere():
#     pass


# cpdef Vector vector_hemisphere_uniform():
#     pass


cpdef Vector vector_hemisphere_cosine():
    """
    Generates a cosine- weighted random vector on a unit hemisphere.

    The hemisphere is aligned along the z-axis - the plane that forms the
    hemisphere based lies in the x-y plane.

    :return: A unit Vector.
    """

    cdef Point p = point_disk()
    return new_vector(p.x, p.y, sqrt(max(0, 1 - p.x*p.x - p.y*p.y)))

