from libc.math cimport sqrt
cimport cython


cdef class StokesVector:

    cdef readonly double i, q, u, v

    # todo: complete docstring
    def __init__(self, double i=0, double q=0, double u=0, double v=0):
        """
        All orientations are defined looking down the beam in the direction of propagation.

            i is the total light intensity
            q and u describe the angle of linear polarisation
            v is the handedness of the circular polarisation
                v > 0: Right-handed polarisation
                v < 0: Left-handed polarisation

        Each component of the Stoke's vector has units of spectral radiant intensity (W/m2/str/nm).

        The Stoke's vector in Raysect is defined such that:

            * horizontal linear polarisation aligns with the x axis
            * vertical linear polarisation aligns with the y axis
            * beam propagation is along the z axis
            * right hand circular light rotates counter-clockwise looking along the beam in the direction of propagation

        """

        if i < 0:
            raise ValueError('The total radiance of the Stoke\'s vector cannot be less than 0.')

        polarised_radiance = sqrt(self.q*self.q + self.u*self.u + self.v*self.v)
        if polarised_radiance > i:
            raise ValueError('The polarised radiance fraction cannot exceed the total radiance.')

        self.i = i
        self.q = q
        self.u = u
        self.v = v

    @cython.cdivision(True)
    cpdef double polarised_fraction(self):
        """
        Returns the degree of polarisation.
        
        A value in the range [0, 1] is returned that indicates the degree of
        polarisation. A value of zero means the light is unpolarised. A value
        of one means the light is entirely polarised.
        
        :returns: Fraction of polarisation.   
        """

        if self.i == 0.0:
            return 0.0

        return sqrt(self.q*self.q + self.u*self.u + self.v*self.v) / self.i

    @cython.cdivision(True)
    cpdef double linear_fraction(self):
        """
        Returns the degree of linear polarisation.
        
        A value in the range [0, 1] is returned that indicates the degree of
        linear polarisation. A value of zero means there is no linear
        polarised component. A value of one means the light is entirely
        linear polarised.
        
        :returns: Fraction of linear polarisation.   
        """

        if self.i == 0.0:
            return 0.0

        return sqrt(self.q*self.q + self.u*self.u) / self.i

    @cython.cdivision(True)
    cpdef double circular_fraction(self):
        """
        Returns the degree of circular polarisation.
        
        A value in the range [0, 1] is returned that indicates the degree of
        circular polarisation. A value of zero means there is no circular
        polarised component. A value of one means the light is entirely
        circular polarised.
        
        :returns: Fraction of circular polarisation.   
        """

        if self.i == 0.0:
            return 0.0

        return self.v / self.i

    def __getstate__(self):
        """Encodes state for pickling."""

        return self.i, self.q, self.u, self.v

    def __setstate__(self, state):
        """Decodes state for pickling."""

        self.i = state[0]
        self.q = state[1]
        self.u = state[2]
        self.v = state[3]

    # todo: implement __mul__ with both scalar and mueller
    # todo: implement mul with scalar
    # todo: implement transform() with mueller matrix
    # todo: implement stokes vector addition