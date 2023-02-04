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

import numpy as np
cimport numpy as np
from raysect.core.math cimport Point3D, Vector3D, AffineMatrix3D
from raysect.core.math.random cimport uniform, vector_sphere, vector_cone_uniform, vector_hemisphere_cosine
from raysect.core.math.cython cimport find_index, rotate_basis
from libc.math cimport M_PI, M_1_PI, asin, acos, sqrt
cimport cython


cdef class _TargettedSampler:

    def __init__(self, object targets):
        """
        Targets is a list of tuples describing the target spheres and their weighting.

        Each target tuple consists of (Point3D sphere_centre, double sphere_radius, double weight).

        :param list targets: A list of tuples describing spheres for targetted sampling.
        """

        self._targets = tuple(targets)
        self._total_weight = 0
        self._cdf = None

        self._validate_targets()
        self._calculate_cdf()

    def __getstate__(self):
        state = self._total_weight, self._targets, self._cdf

    def __setstate__(self, state):
        self._total_weight, self._targets, self._cdf = state
        self._cdf_mv = self._cdf

    def __reduce__(self):
        return self.__new__, (self.__class__, ), self.__getstate__()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef object _validate_targets(self):

        cdef:
            tuple target
            double radius, weight

        if len(self._targets) < 1:
            raise ValueError('List of targets must contain at least one target sphere.')

        for target in self._targets:
            _, radius, weight = target

            if radius <= 0:
                raise ValueError('Target sphere radius must be greater than zero.')

            if weight <= 0:
                raise ValueError('Target weight must be greater than zero.')

    def __call__(self, Point3D point, object samples=None, bint pdf=False):
        """
        If samples is not provided, returns a single Vector3D sample from
        the distribution. If samples is set to a value then a number of
        samples equal to the value specified is returned in a list.

        If pdf is set to True the Vector3D sample is returned inside a tuple
        with its associated pdf value as the second element.

        :param int samples: Number of points to generate (default=None).
        :param bool pdf: Toggle for returning associated sample pdfs (default=False).
        :return: A Vector3D, tuple or list of Vector3D objects.
        """

        if samples:
            samples = int(samples)
            if samples <= 0:
                raise ValueError("Number of samples must be greater than 0.")
            if pdf:
                return self.samples_with_pdfs(point, samples)
            return self.samples(point, samples)
        else:
            if pdf:
                return self.sample_with_pdf(point)
            return self.sample(point)

    cpdef double pdf(self, Point3D point, Vector3D sample):
        """
        Calculates the value of the PDF for the specified sample point and direction.

        :param Point3D point: The point from which to sample.
        :param Vector3D sample: The sample direction.
        :rtype: double
        """

        raise NotImplementedError('Virtual method pdf() has not been implemented.')

    cdef Vector3D sample(self, Point3D point):
        """
        Generate a single sample.

        If the pdf is required please see sample_with_pdf().

        :param Point3D point: The point from which to sample.
        :return: The vector along which to sample.
        :rtype: Vector3D
        """

        raise NotImplementedError('Virtual method sample() has not been implemented.')

    cdef tuple sample_with_pdf(self, Point3D point):
        """
        Generates a single sample with its associated pdf.

        Returns a tuple with the sample point as the first element and pdf value as
        the second element.

        Obtaining a sample with its pdf is generally more efficient than requesting the sample and then
        its pdf in a subsequent call since some of the calculation is common between the two steps.

        :rtype: tuple
        """

        cdef:
            Vector3D sample_direction
            double pdf

        sample_direction = self.sample(point)
        pdf = self.pdf(point, sample_direction)

        return sample_direction, pdf

    cdef list samples(self, Point3D point, int samples):
        """
        Generates a list of samples.

        If pdfs are required please see samples_with_pdfs().

        :param int samples: Number of points to generate.
        :rtype: list
        """

        cdef list results
        cdef int i

        results = []
        for i in range(samples):
            results.append(self.sample(point))
        return results

    cdef list samples_with_pdfs(self, Point3D point, int samples):
        """
        Generates a list of tuples containing samples and their associated pdfs.

        Each sample is a tuple with the sample point as the first element and pdf value as
        the second element.

        Obtaining samples with pdfs is generally more efficient than requesting samples and then
        the pdf in a subsequent call since some of the calculation is common between the two steps.

        :param int samples: Number of points to generate.
        :rtype: list
        """

        cdef list results
        cdef int i

        results = []
        for i in range(samples):
            results.append(self.sample_with_pdf(point))
        return results

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef object _calculate_cdf(self):
        """
        Calculate the cumulative distribution function for the sphere weights.

        Stores an array with length equal to the number of spheres. At each
        point in the array the normalised cumulative weighting is stored.
        """

        cdef:
            int count, index
            tuple sphere_data
            double weight

        count = len(self._targets)

        # create empty array and acquire a memory view for fast access
        self._cdf = np.zeros(count)
        self._cdf_mv = self._cdf

        # accumulate cdf and total weight
        for index, sphere_data in enumerate(self._targets):
            _, radius, weight = sphere_data
            if index == 0:
                self._cdf_mv[index] = weight
            else:
                self._cdf_mv[index] = self._cdf_mv[index - 1] + weight

        # normalise
        self._total_weight = self._cdf_mv[count - 1]
        for index in range(count - 1):
            self._cdf_mv[index] /= self._total_weight

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef tuple _pick_sphere(self):
        """
        Find the important primitive bounding sphere corresponding to a uniform random number.
        """

        cdef int index

        # due to the CDF not starting at zero, using find_index means that the result is offset by 1 index point.
        index = find_index(self._cdf_mv, uniform()) + 1
        return self._targets[index]


cdef class TargettedHemisphereSampler(_TargettedSampler):
    """
    Generates vectors on a hemisphere targetting a set of target spheres.

    This sampler takes a list of spheres and corresponding weighting factors.
    To generate a sample a sphere is randomly selected according the
    distribution of the sphere weights and launches a ray at the solid angle
    subtended by the target sphere.

    If the targetted sphere intersects with the hemisphere's base plane, lies
    behind the plane, or the origin point from which samples are generated lies
    inside the target sphere, a sample is randomly selected from the full hemisphere
    using a cosine weighted distribution.

    The hemisphere is aligned so the hemisphere base lies on the x-y plane and
    the points along the +ve z axis.
    """

    @cython.cdivision(True)
    cpdef double pdf(self, Point3D point, Vector3D sample):
        """
        Calculates the value of the PDF for the specified sample point and direction.

        :param Point3D point: The point from which to sample.
        :param Vector3D sample: The sample direction.
        :rtype: double
        """

        cdef:
            double weight, selection_weight
            double pdf, pdf_all, sphere_radius
            double  distance, solid_angle, angular_radius_cos, t
            Point3D sphere_centre
            Vector3D cone_axis
            AffineMatrix3D rotation

        # Sample is beneath the hemisphere plane
        if sample.z < 0.0:
            return 0.0

        # assemble pdf
        pdf_all = 0
        for sphere_centre, sphere_radius, weight in self._targets:

            cone_axis = point.vector_to(sphere_centre)
            distance = cone_axis.get_length()

            # is point inside sphere?
            if distance == 0 or distance < sphere_radius:
                pdf = M_1_PI * sample.z

            else:

                # calculate the angular radius and solid angle projection of the sphere
                t = sphere_radius / distance
                angular_radius = asin(t)

                # Tests direction angle is always above cone angle:
                # We have yet to derive the partial projected area of a sphere intersecting the horizon
                # for now we will just fall back to hemisphere sampling, even if this is inefficient.
                # This test also implicitly checks if the sphere direction lies in the hemisphere as the
                # angular_radius cannot be less than zero
                # todo: calculate projected area of a cut sphere to improve sampling
                sphere_angle = asin(sample.z)
                if angular_radius >= sphere_angle:
                    pdf = M_1_PI * sample.z

                else:

                    # calculate cosine of angular radius of cone
                    angular_radius_cos = sqrt(1 - t * t)

                    # does the direction lie inside the cone of projection
                    cone_axis = cone_axis.normalise()
                    if sample.dot(cone_axis) < angular_radius_cos:
                        # no contribution, outside code of projection
                        continue

                    # calculate pdf
                    solid_angle = 2 * M_PI * (1 - angular_radius_cos)
                    pdf = 1 / solid_angle

            # add contribution to pdf
            selection_weight = weight / self._total_weight
            pdf_all += selection_weight * pdf

        return pdf_all

    @cython.cdivision(True)
    cdef Vector3D sample(self, Point3D point):
        """
        Generate a single sample.

        If the pdf is required please see sample_with_pdf().

        :param Point3D point: The point from which to sample.
        :return: The vector along which to sample.
        :rtype: Vector3D
        """

        # calculate projection of sphere (a disk) as seen from origin point and
        # generate a random direction towards that projection

        cdef:
            double sphere_radius, weight, distance, angular_radius
            Point3D sphere_centre
            Vector3D direction, sample
            AffineMatrix3D rotation

        sphere_centre, sphere_radius, weight = self._pick_sphere()

        direction = point.vector_to(sphere_centre)
        distance = direction.get_length()

        # is point inside sphere?
        if distance == 0 or distance < sphere_radius:
            # the point lies inside the sphere, sample random direction from full sphere
            return vector_hemisphere_cosine()

        # calculate the angular radius and solid angle projection of the sphere
        angular_radius = asin(sphere_radius / distance)

        # Tests direction angle is always above cone angle:
        # We have yet to derive the partial projected area of a sphere intersecting the horizon
        # for now we will just fall back to hemisphere sampling, even if this is inefficient.
        # This test also implicitly checks if the sphere direction lies in the hemisphere as the
        # angular_radius cannot be less than zero
        # todo: calculate projected area of a cut sphere to improve sampling
        direction = direction.normalise()
        sphere_angle = asin(direction.z)
        if angular_radius >= sphere_angle:
            return vector_hemisphere_cosine()

        # sample a vector from a cone of half angle equal to the angular radius
        sample = vector_cone_uniform(angular_radius * 180 / M_PI)

        # rotate cone to lie along vector from observation point to sphere centre
        rotation = rotate_basis(direction, direction.orthogonal())
        return sample.transform(rotation)


cdef class TargettedSphereSampler(_TargettedSampler):
    """
    Generates vectors targetting a set of target spheres.

    This sampler takes a list of spheres and corresponding weighting factors.
    To generate a sample a sphere is randomly selected according the
    distribution of the sphere weights and launches a ray at the solid angle
    subtended by the target sphere.

    If the origin point, from which samples are generated, lies inside the
    target sphere. A random sample is generated by uniformly sampling over the
    full sphere.
    """

    @cython.cdivision(True)
    cpdef double pdf(self, Point3D point, Vector3D sample):
        """
        Calculates the value of the PDF for the specified sample point and direction.

        :param Point3D point: The point from which to sample.
        :param Vector3D sample: The sample direction.
        :rtype: double
        """

        cdef:
            double weight, selection_weight
            double pdf, pdf_all, sphere_radius
            double  distance, solid_angle, angular_radius_cos, t
            Point3D sphere_centre
            Vector3D cone_axis
            AffineMatrix3D rotation

        pdf_all = 0
        for sphere_centre, sphere_radius, weight in self._targets:

            cone_axis = point.vector_to(sphere_centre)
            distance = cone_axis.get_length()

            # is point inside sphere?
            if distance == 0 or distance < sphere_radius:

                # the point lies inside the sphere, the projection is a full sphere
                solid_angle = 4 * M_PI

            else:

                # calculate cosine of angular radius of cone
                t = sphere_radius / distance
                angular_radius_cos = sqrt(1 - t * t)

                # does the direction lie inside the cone of projection
                cone_axis = cone_axis.normalise()
                if sample.dot(cone_axis) < angular_radius_cos:
                    # no contribution, outside code of projection
                    continue

                # calculate solid angle
                solid_angle = 2 * M_PI * (1 - angular_radius_cos)

            # calculate probability
            pdf = 1 / solid_angle
            selection_weight = weight / self._total_weight

            # add contribution to pdf
            pdf_all += selection_weight * pdf

        return pdf_all

    @cython.cdivision(True)
    cdef Vector3D sample(self, Point3D point):
        """
        Generate a single sample.

        If the pdf is required please see sample_with_pdf().

        :param Point3D point: The point from which to sample.
        :return: The vector along which to sample.
        :rtype: Vector3D
        """

        # calculate projection of sphere (a disk) as seen from origin point and
        # generate a random direction towards that projection

        cdef:
            double sphere_radius, weight, distance, angular_radius
            Point3D sphere_centre
            Vector3D direction, sample
            AffineMatrix3D rotation

        sphere_centre, sphere_radius, weight = self._pick_sphere()

        direction = point.vector_to(sphere_centre)
        distance = direction.get_length()

        # is point inside sphere?
        if distance == 0 or distance < sphere_radius:
            # the point lies inside the sphere, sample random direction from full sphere
            return vector_sphere()

        # calculate the angular radius and solid angle projection of the sphere
        angular_radius = asin(sphere_radius / distance)

        # sample a vector from a cone of half angle equal to the angular radius
        sample = vector_cone_uniform(angular_radius * 180 / M_PI)

        # rotate cone to lie along vector from observation point to sphere centre
        direction = direction.normalise()
        rotation = rotate_basis(direction, direction.orthogonal())
        return sample.transform(rotation)

