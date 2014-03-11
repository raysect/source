# cython: language_level=3

# Copyright (c) 2014, Dr Alex Meakins, Raysect Project
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

cdef class MaterialRGB(Material):

    cpdef SurfaceRGB evaluate_surface(self, World world, Ray ray, Primitive primitive, Point hit_point,
                                            bint exiting, Point inside_point, Point outside_point,
                                            Normal normal, AffineMatrix to_local, AffineMatrix to_world):

        return NotImplemented

    cpdef VolumeRGB evaluate_volume(self, World world, Ray ray, Point entry_point, Point exit_point,
                                         AffineMatrix to_local, AffineMatrix to_world):

        return NotImplemented


cdef class VolumeEmissionIntegrator(MaterialRGB):

    def __init__(self, double step = 0.01):

        self.step = step

    cpdef SurfaceRGB evaluate_surface(self, World world, Ray ray, Primitive primitive, Point hit_point,
                                            bint exiting, Point inside_point, Point outside_point,
                                            Normal normal, AffineMatrix to_local, AffineMatrix to_world):

        cdef Point origin
        cdef RayRGB daughter_ray

        # are we entering or leaving surface?
        if exiting:

            # ray leaving surface, use outer point for ray origin
            origin = outside_point.transform(to_world)

        else:

            # ray entering surface, use inner point for ray origin
            origin = inside_point.transform(to_world)

        daughter_ray = ray.spawn_daughter(origin, ray.direction)

        return SurfaceRGB(daughter_ray.trace(world))

    cpdef VolumeRGB evaluate_volume(self, World world, Ray ray, Point entry_point, Point exit_point,
                                         AffineMatrix to_local, AffineMatrix to_world):

        cdef RGB intensity, glow, glow_previous
        cdef Point entry, exit
        cdef Vector direction
        cdef double length, t

        intensity = RGB(0, 0, 0)

        # convert entry and exit to local space
        entry = entry_point.transform(to_local)
        exit = exit_point.transform(to_local)

        # ray direction and path length
        direction = entry.vector_to(exit)
        length = direction.length
        direction = direction.normalise()

        # numerical integration
        glow_previous = self.emission_function(entry)
        t = self.step
        while(t <= length):

            glow = self.emission_function(new_point(entry.x + t * direction.x,
                                                    entry.y + t * direction.y,
                                                    entry.z + t * direction.z))

            # trapizium rule integration
            intensity.r += 0.5 * self.step * (glow.r + glow_previous.r)
            intensity.g += 0.5 * self.step * (glow.g + glow_previous.g)
            intensity.b += 0.5 * self.step * (glow.b + glow_previous.b)

            glow_previous = glow
            t += self.step

        glow = self.emission_function(exit)

        # trapizium rule integration of remainder
        t -= self.step
        intensity.r += 0.5 * (length - t) * (glow.r + glow_previous.r)
        intensity.g += 0.5 * (length - t) * (glow.g + glow_previous.g)
        intensity.b += 0.5 * (length - t) * (glow.b + glow_previous.b)

        return VolumeRGB(intensity, RGB(0, 0, 0))

    cpdef RGB emission_function(self, Point point):

        return NotImplemented


cdef class Glow(MaterialRGB):

    def __init__(self, RGB colour = RGB(0, 0, 0)):

        self.colour = colour

    cpdef SurfaceRGB evaluate_surface(self, World world, Ray ray, Primitive primitive, Point hit_point,
                                            bint exiting, Point inside_point, Point outside_point,
                                            Normal normal, AffineMatrix to_local, AffineMatrix to_world):

        cdef Point origin
        cdef RayRGB daughter_ray

        # are we entering or leaving surface?
        if exiting:

            # ray leaving surface, use outer point for ray origin
            origin = outside_point.transform(to_world)

        else:

            # ray entering surface, use inner point for ray origin
            origin = inside_point.transform(to_world)

        daughter_ray = ray.spawn_daughter(origin, ray.direction)

        return SurfaceRGB(daughter_ray.trace(world))

    cpdef VolumeRGB evaluate_volume(self, World world, Ray ray, Point entry_point, Point exit_point,
                                         AffineMatrix to_local, AffineMatrix to_world):

        cdef double length

        # calculate ray length
        length = entry_point.vector_to(exit_point).length

        # calculate light contribution (defined as intensity per meter)
        return VolumeRGB(RGB(self.colour.r * length,
                             self.colour.g * length,
                             self.colour.b * length),
                         RGB(0, 0, 0))


cdef class GlowGaussian(VolumeEmissionIntegrator):

    def __init__(self, RGB colour = RGB(1, 1, 1), double sigma = 0.2, double step = 0.01):

        super().__init__(step)

        self.colour = colour
        self.sigma = sigma

    property sigma:

        def __get__(self):

            return self.sigma

        def __set__(self, double sigma):

            self._sigma = sigma
            self._denominator = 0.5 / (sigma * sigma)

    cpdef RGB emission_function(self, Point point):

        scale = exp(-self._denominator * (point.x * point.x + point.y * point.y + point.z * point.z))
        return RGB(self.colour.r * scale, self.colour.g * scale, self.colour.b * scale)


cdef class GlowGaussianBeam(VolumeEmissionIntegrator):

    def __init__(self, RGB colour = RGB(1, 1, 1), double sigma = 0.2, double step = 0.01):

        super().__init__(step)

        self.colour = colour
        self.sigma = sigma

    property sigma:

        def __get__(self):

            return self.sigma

        def __set__(self, double sigma):

            self._sigma = sigma
            self._denominator = 0.5 / (sigma * sigma)

    cpdef RGB emission_function(self, Point point):

        scale = exp(-self._denominator * (point.x * point.x + point.y * point.y))
        return RGB(self.colour.r * scale, self.colour.g * scale, self.colour.b * scale)


cdef class GlowBeams(VolumeEmissionIntegrator):

    def __init__(self, RGB colour = RGB(1, 1, 1), double scale = 0.1, double step = 0.02):

        super().__init__(step)

        self.colour = colour
        self.scale = scale

    property scale:

        def __get__(self):

            return self._scale

        def __set__(self, double scale):

            self._scale = scale
            self._multiplier = 2 * pi / scale

    cpdef RGB emission_function(self, Point point):

        cdef double sx, sy, scale

        sx = sin(self._multiplier * point.x)
        sy = sin(self._multiplier * point.y)

        scale = (sx * sx) + (sy * sy)
        scale = scale * scale * scale * scale

        return RGB(self.colour.r * scale, self.colour.g * scale, self.colour.b * scale)


cdef class Checkerboard(MaterialRGB):

    def __init__(self):

        self._scale = 1.0
        self._rscale = 1.0
        self.colourA = RGB(0.06, 0.06, 0.08)
        self.colourB = RGB(0.12, 0.12, 0.16)

    property scale:

        def __get__(self):

            return self._scale

        def __set__(self, double v):

            self._scale = v
            self._rscale = 1 / v

    cpdef SurfaceRGB evaluate_surface(self, World world, Ray ray, Primitive primitive, Point hit_point,
                                            bint exiting, Point inside_point, Point outside_point,
                                            Normal normal, AffineMatrix to_local, AffineMatrix to_world):

        cdef bint v

        v = False

        # generate check pattern
        v = self._flip(v, hit_point.x)
        v = self._flip(v, hit_point.y)
        v = self._flip(v, hit_point.z)

        # apply "colours"
        return SurfaceRGB(RGB(self.colourA.r * v + self.colourB.r * (not v),
                              self.colourA.g * v + self.colourB.g * (not v),
                              self.colourA.b * v + self.colourB.b * (not v)))

    cpdef VolumeRGB evaluate_volume(self, World world, Ray ray, Point entry_point, Point exit_point,
                                         AffineMatrix to_local, AffineMatrix to_world):

        return VolumeRGB(RGB(0, 0, 0), RGB(0, 0, 0))

    cdef inline bint _flip(self, bint v, double p):

        # round to avoid numerical precision issues (hardcoded to 1nm min scale!)
        p = round(p, 9)

        # generates check pattern from [0, inf]
        if abs(self._rscale * p) % 2 >= 1.0:

            v = not v

        # invert pattern for negative
        if p < 0:

            v = not v

        return v


#class Glass(Material):

    #def __init__(self, index = 1.52):

        #self.index = index
        #self.transmission = array([0.994, 0.996, 0.993])
        #self.transmission_depth = 0.025
        #self.cutoff = 1e-6

    #def evaluate_surface(self, world, ray, primitive, point, normal):

        ## convert hit point to world space
        #w_point = point.transform(primitive.to_world())

        ## convert ray direction normal to local coordinates
        #incident = ray.direction.transform(primitive.to_local())

        ## calculate cosine of angle between incident and normal
        #c1 = -normal.dot(incident)

        ## are we entering or leaving material - calculate refractive change
        #if c1 > 0:

            ## entering material
            #n1 = 1.0
            #n2 = self.index

        #else:

            ## leaving material
            #n1 = self.index
            #n2 = 1.0

        #gamma = n1 / n2

        ## calculate square of cosone of angle between transmitted ray and normal
        #c2s = 1 - (gamma*gamma) * (1 - c1*c1)

        ## check for total internal reflection
        #if c2s <= 0:

            ## total internal reflection
            #reflected = Normal(incident + 2 * c1 * normal)

            ## convert reflected ray to world space
            #w_reflected = reflected.transform(primitive.to_world())

            ## spawn reflected ray and trace
            #r_reflected = ray.spawn_daughter(w_point, w_reflected)
            #return r_reflected.trace(world)

        #else:

            ## calculate reflected and transmitted ray normals
            #reflected = Normal(incident + 2 * c1 * normal)
            #if c1 >=0:
                #transmitted = Normal(gamma * incident + (gamma * c1 - sqrt(c2s)) * normal)
            #else:
                #transmitted = Normal(gamma * incident + (gamma * c1 + sqrt(c2s)) * normal)

            ## convert reflected and transmitted rays to world space
            #w_reflected = reflected.transform(primitive.to_world())
            #w_transmitted = transmitted.transform(primitive.to_world())

            ## spawn reflected and transmitted rays
            #r_reflected = ray.spawn_daughter(w_point, w_reflected)
            #r_transmitted = ray.spawn_daughter(w_point, w_transmitted)

            ## calculate fresnel reflection and transmission
            #(reflectivity, transmission) = self._fresnel(c1, -normal.dot(transmitted), n1, n2)

            ## trace rays and return results
            #v = 0.0
            #if reflectivity > self.cutoff:
                #v = reflectivity * r_reflected.trace(world)

            #if transmission > self.cutoff:
                #v = v + transmission * r_transmitted.trace(world)

            #return v

    #def _fresnel(self, ci, ct, n1, n2):

        #r = 0.5 * (((n1*ci - n2*ct) / (n1*ci + n2*ct))**2 + ((n1*ct - n2*ci) / (n1*ct + n2*ci))**2)
        #t = 1 - r
        #return (r,t)

    #def evaluate_volume(self, world, ray, primitive, point1, point2):

        #d = point1.vector_to(point2).length
        #t = self.transmission**(d / float(self.transmission_depth))
        #return (array([0.0, 0.0, 0.0]), t)
