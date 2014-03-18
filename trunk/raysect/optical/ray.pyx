# cython: language_level=3

#Copyright (c) 2014, Dr Alex Meakins, Raysect Project
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

# cython doesn't have a built-in infinity constant, this compiles to +infinity
DEF INFINITY = 1e999

cdef class Waveband:
    """
    waveband: [min_wavelength, max_wavelength)
    """

    def __init__(self, double min_wavelength, double max_wavelength):

        if min_wavelength <= 0.0 or max_wavelength <= 0.0:

            raise ValueError("Wavelength can not be less than or equal to zero.")

        if min_wavelength >= max_wavelength:

            raise ValueError("Minimum wavelength can not be greater or eaual to the maximum wavelength.")

        self._min_wavelength = min_wavelength
        self._max_wavelength = max_wavelength

    property min_wavelength:

        def __get__(self):

            return self._min_wavelength

        def __set__(self, double wavelength):

            if wavelength <= 0.0:

                raise ValueError("Wavelength can not be less than or equal to zero.")

            if wavelength >= self._max_wavelength:

                raise ValueError("Minimum wavelength can not be greater than or equal to the maximum wavelength.")

            self._min_wavelength = wavelength

    property max_wavelength:

        def __get__(self):

            return self._max_wavelength

        def __set__(self, double wavelength):

            if wavelength <= 0.0:

                raise ValueError("Wavelength can not be less than or equal to zero.")

            if self._min_wavelength >= wavelength:

                raise ValueError("Maximum wavelength can not be less than or equal to the minimum wavelength.")

            self._max_wavelength = wavelength

    cpdef Waveband copy(self):

        return new_waveband(self._min_wavelength, self._max_wavelength)

    cdef inline double get_min_wavelength(self):

        return self._min_wavelength

    cdef inline double get_max_wavelength(self):

        return self._max_wavelength


cdef class RayResponce:

    #TODO: WRITE ME

    pass


cdef class OpticalRay(Ray):

    def __init__(self,
                 Point origin = Point([0,0,0]),
                 Vector direction = Vector([0,0,1]),
                 list wavebands = list(),
                 double refraction_wavelength = 550,
                 double max_distance = INFINITY):

        super().__init__(origin, direction, max_distance)

        self.primary_ray = self
        self.refraction_wavelength = refraction_wavelength
        self.wavebands = wavebands
        self.cache_valid = False

    property refraction_wavelength:

        def __get__(self):

            return self._refraction_wavelength

        def __set__(self, double wavelength):

            if wavelength <= 0:

                raise ValueError("Wavelength can not be less than or equal to zero.")

            self._refraction_wavelength = wavelength

    property wavebands:

        def __get__(self):

            # return a copy to ensure the users cannot change state of waveband
            # objects via externally held references - the ray must be able to
            # track changes to the wavebands and inform the materials when their
            # caches are invalidated

            cdef list waveband_list
            cdef Waveband waveband

            waveband_list = list()
            for waveband in self._wavebands:

                waveband_list.append(waveband.copy())

            return waveband_list

        def __set__(self, list wavebands not None):

            # copy list to ensure the users cannot change state of waveband
            # objects via externally held references - the ray must be able to
            # track changes to the wavebands and inform the materials when their
            # caches are invalidated

            self._wavebands = list()
            for waveband in wavebands:

                self._wavebands.append((<Waveband?> waveband).copy())

            self.cache_valid = False

    def __getitem__(self, int index):

        # return a copy to ensure the users cannot change state of waveband
        # objects via externally held references - the ray must be able to
        # track changes to the wavebands and inform the materials when their
        # caches are invalidated

        return (<Waveband> self._wavebands[index]).copy()

    def __setitem__(self, int index, Waveband waveband not None):

        # copy waveband to ensure the users cannot change state of waveband
        # objects via externally held references - the ray must be able to
        # track changes to the wavebands and inform the materials when their
        # caches are invalidated

        self._wavebands[index] = waveband.copy()
        self.cache_valid = False

    cpdef append_waveband(self, Waveband waveband):

        if waveband is None:

            raise TypeError("A Waveband object is required, arguement cannot be None.")

        # copy waveband to ensure the users cannot change state of waveband
        # objects via externally held references - the ray must be able to
        # track changes to the wavebands and inform the materials when their
        # caches are invalidated

        self._wavebands.append(waveband.copy())
        self.cache_valid = False

    cpdef object trace(self, World world):

        #TODO: WRITE ME

        self.cache_valid = True

        return RayResponce()

    cpdef Ray spawn_daughter(self, Point origin, Vector direction):

        #TODO: WRITE ME

        return NotImplemented

    cdef inline double get_refraction_wavelength(self):

        return self._refraction_wavelength

    cdef inline int get_waveband_count(self):

        return len(self._wavebands)

    cdef inline Waveband get_waveband(self, int index):

        return self._wavebands[index]


def monocromatic_ray(origin, direction, min_wavelength, max_wavelength, max_distance = INFINITY):

    return OpticalRay(origin,
                      direction,
                      [Waveband(min_wavelength, max_wavelength)],
                      0.5 * (min_wavelength + max_wavelength),
                      max_distance)


def polycromatic_ray(origin, direction, min_wavelength, max_wavelength, steps, max_distance = INFINITY):

    wavebands = list()

    delta_wavelength = (max_wavelength - min_wavelength) / steps

    for index in range(0, steps):

        wavebands.append(Waveband(min_wavelength + delta_wavelength * index,
                                  min_wavelength + delta_wavelength * (index + 1)))

    return OpticalRay(origin,
                      direction,
                      wavebands,
                      0.5 * (min_wavelength + max_wavelength),
                      max_distance)
