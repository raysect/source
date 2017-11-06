# cython: language_level=3

# Copyright (c) 2014-2017, Dr Alex Meakins, Raysect Project
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

from raysect.core cimport Point3D, Vector3D


# todo: docstrings need updating
cdef class TargettedHemisphereSampler:

    cdef:
        double _targetted_path_prob
        list _targets

    def __init__(self, list targets, targetted_path_prob=None):
        """
        Targets is a list of tuples containing the following (Point3D sphere_origin, double sphere_radius, double weight).

        :param list targets: a list of tuples describing spheres for targetted sampling.
        """
        self.targetted_path_prob = targetted_path_prob or 0.9
        self._targets = targets

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

    @property
    def targetted_path_prob(self):
        return self._targetted_path_prob

    @targetted_path_prob.setter
    def targetted_path_prob(self, double value):

        if value < 0 or value > 1:
            raise ValueError("Targeted path probability must lie in the range [0, 1].")
        self._targetted_path_prob = value

    cpdef double pdf(self, Point3D point, Vector3D sample):
        """
        Generates a pdf for a given sample value.

        Vectors *must* be normalised.

        :param Vector3D sample: The sample point at which to get the pdf.
        :rtype: float
        """
        raise NotImplemented("The method pdf() is not implemented for this sampler.")

    cdef Vector3D sample(self, Point3D point):
        """
        Generate a single sample.

        If the pdf is required please see sample_with_pdf().

        :rtype: Vector3D
        """
        raise NotImplemented("The method sample() is not implemented for this sampler.")

    cdef tuple sample_with_pdf(self, Point3D point):
        """
        Generates a single sample with its associated pdf.

        Returns a tuple with the sample point as the first element and pdf value as
        the second element.

        Obtaining a sample with its pdf is generally more efficient than requesting the sample and then
        its pdf in a subsequent call since some of the calculation is common between the two steps.

        :rtype: tuple
        """
        raise NotImplemented("The method pdf() is not implemented for this sampler.")

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


# todo: docstrings need updating
cdef class TargettedSphereSampler:

    cdef:
        double _targetted_path_prob
        list _targets

    def __init__(self, list targets, targetted_path_prob=None):
        """
        Targets is a list of tuples containing the following (Point3D sphere_origin, double sphere_radius, double weight).

        :param list targets: a list of tuples describing spheres for targetted sampling.
        """
        self.targetted_path_prob = targetted_path_prob or 0.9
        self._targets = targets

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

    @property
    def targetted_path_prob(self):
        return self._targetted_path_prob

    @targetted_path_prob.setter
    def targetted_path_prob(self, double value):

        if value < 0 or value > 1:
            raise ValueError("Targetted path probability must lie in the range [0, 1].")
        self._targetted_path_prob = value

    cpdef double pdf(self, Point3D point, Vector3D sample):
        """
        Generates a pdf for a given sample value.

        Vectors *must* be normalised.

        :param Vector3D sample: The sample point at which to get the pdf.
        :rtype: float
        """
        raise NotImplemented("The method pdf() is not implemented for this sampler.")

    cdef Vector3D sample(self, Point3D point):
        """
        Generate a single sample.

        If the pdf is required please see sample_with_pdf().

        :rtype: Vector3D
        """
        raise NotImplemented("The method sample() is not implemented for this sampler.")

    cdef tuple sample_with_pdf(self, Point3D point):
        """
        Generates a single sample with its associated pdf.

        Returns a tuple with the sample point as the first element and pdf value as
        the second element.

        Obtaining a sample with its pdf is generally more efficient than requesting the sample and then
        its pdf in a subsequent call since some of the calculation is common between the two steps.

        :rtype: tuple
        """
        raise NotImplemented("The method pdf() is not implemented for this sampler.")

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
