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

from raysect.core.math.random cimport probability
from raysect.optical cimport new_affinematrix3d


cdef class Material(CoreMaterial):
    """
    Base class for optical material classes.

    Derived classes must implement the evaluate_surface() and evaluate_volume() methods.
    """

    def __init__(self):
        super().__init__()
        self._importance = 0.0

    @property
    def importance(self):
        """
        Importance sampling weight for this material.

        Only effective if importance sampling is turned on.

        :rtype: float
        """
        return self._importance

    @importance.setter
    def importance(self, value):
        if value < 0:
            raise ValueError("Material sampling importance cannot be less than zero.")
        self._importance = value
        self.notify_material_change()

    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Primitive primitive, Point3D hit_point,
                                    bint exiting, Point3D inside_point, Point3D outside_point,
                                    Normal3D normal, AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world,
                                    Intersection intersection):
        """
        Virtual method for evaluating the spectrum at a material surface.

        :param World world: The world scenegraph belonging to this material.
        :param Ray ray: The ray incident at the material surface.
        :param Primitive primitive: The geometric shape the holds this material
          (i.e. mesh, cylinder, etc.).
        :param Point3D hit_point: The point where the ray is incident on the
          primitive surface.
        :param bool exiting: Boolean toggle indicating if this ray is exiting or
          entering the material surface (True means ray is exiting).
        :param Point3D inside_point:
        :param Point3D outside_point:
        :param Normal3D normal: The surface normal vector at location of hit_point.
        :param AffineMatrix3D world_to_primitive: Affine matrix defining transformation
          from world space to local primitive space.
        :param AffineMatrix3D primitive_to_world: Affine matrix defining transformation
          from local primitive space to world space.
        :param Intersection intersection: The full ray-primitive intersection object.
        """
        raise NotImplementedError("Material virtual method evaluate_surface() has not been implemented.")

    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world, Ray ray, Primitive primitive,
                                   Point3D start_point, Point3D end_point,
                                   AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):
        """
        Virtual method for evaluating the spectrum emitted/absorbed along the rays trajectory
        through a material surface.


        :param Spectrum spectrum: The spectrum already accumulated along the ray path.
          Don't overwrite this array, add the materials emission/absorption to the existing
          spectrum.
        :param World world: The world scenegraph belonging to this material.
        :param Ray ray: The ray incident at the material surface.
        :param Primitive primitive: The geometric shape the holds this material
          (i.e. mesh, cylinder, etc.).
        :param Point3D start_point: The starting point of the ray's trajectory
          through the material.
        :param Point3D end_point: The end point of the ray's trajectory through
          the material.
        :param AffineMatrix3D world_to_primitive: Affine matrix defining transformation
          from world space to local primitive space.
        :param AffineMatrix3D primitive_to_world: Affine matrix defining transformation
          from local primitive space to world space.
        """
        raise NotImplementedError("Material virtual method evaluate_volume() has not been implemented.")


cdef class NullSurface(Material):
    """
    A base class for materials that have volume properties such as emission/absorption
    but no surface properties (e.g. a plasma). This material will launch a new ray after
    the initial ray has transited the material primitive's volume. evaluate_volume() must be
    implemented by the deriving class.
    """

    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Primitive primitive, Point3D hit_point,
                                    bint exiting, Point3D inside_point, Point3D outside_point,
                                    Normal3D normal, AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world,
                                    Intersection intersection):

        cdef:
            Point3D origin
            Ray daughter_ray

        # are we entering or leaving surface?
        if exiting:
            origin = outside_point.transform(primitive_to_world)
        else:
            origin = inside_point.transform(primitive_to_world)

        daughter_ray = ray.spawn_daughter(origin, ray.direction)

        # do not count null surfaces in ray depth
        daughter_ray.depth -= 1

        # prevent extinction on a null surface
        return daughter_ray.trace(world, keep_alive=True)


cdef class NullVolume(Material):
    """
    A base class for materials that have surface properties such as reflection
    but no volume properties (e.g. a metallic mirror). evaluate_surface() must be
    implemented by the deriving class.
    """

    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world, Ray ray, Primitive primitive,
                                   Point3D start_point, Point3D end_point,
                                   AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        # no volume contribution
        return spectrum


cdef class NullMaterial(Material):
    """
    A perfectly transparent material.

    Behaves as if nothing is present, doesn't perturb the ray trajectories at all.
    Useful for logging ray trajectories and designating areas of interest that may
    not correspond to a physical material (i.e. a slit / aperture).
    """

    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Primitive primitive, Point3D hit_point,
                                    bint exiting, Point3D inside_point, Point3D outside_point,
                                    Normal3D normal, AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world,
                                    Intersection intersection):

        cdef:
            Point3D origin
            Ray daughter_ray

        # are we entering or leaving surface?
        if exiting:
            origin = outside_point.transform(primitive_to_world)
        else:
            origin = inside_point.transform(primitive_to_world)

        daughter_ray = ray.spawn_daughter(origin, ray.direction)

        # do not count null surfaces in ray depth
        daughter_ray.depth -= 1

        # prevent extinction on a null surface
        return daughter_ray.trace(world, keep_alive=True)

    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world, Ray ray, Primitive primitive,
                                   Point3D start_point, Point3D end_point,
                                   AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        # no volume contribution
        return spectrum


# Surface space
#
# to simplify maths:
# normal aligned (flipped) to sit on same side of surface as incoming ray
# incoming ray vector is aligned to point out of the surface
# surface space normal is aligned to lie along +ve Z-axis i.e. Normal3D(0, 0, 1)
#
# The w_reflection_origin and w_transmission_origin points are provided as
# ray launch points. These points are guaranteed to prevent same-surface
# re-intersections. The reflection origin lies on the same side of the
# surface as the incoming ray, the transmission origin lies on the opposite
# side of the surface.
#
# back_face is true if the ray is on the back side of the primitive surface,
# true if on the front side (ie on the side of the primitive surface normal)
cdef class DiscreteBSDF(Material):
    """
    A base class for materials implementing a discrete BSDF.
    """

    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Primitive primitive, Point3D p_hit_point,
                                    bint exiting, Point3D p_inside_point, Point3D p_outside_point,
                                    Normal3D p_normal, AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world,
                                    Intersection intersection):

        cdef:
            Vector3D w_outgoing, s_incoming
            Point3D w_reflection_origin, w_transmission_origin
            AffineMatrix3D world_to_surface, surface_to_world, primitive_to_surface, surface_to_primitive

        # surface space is aligned relative to the incoming ray
        # define ray launch points and orient normal appropriately
        if exiting:

            # ray incident on back face
            w_reflection_origin = p_inside_point.transform(primitive_to_world)
            w_transmission_origin = p_outside_point.transform(primitive_to_world)

            # flip normal
            p_normal = p_normal.neg()

        else:

            # ray incident on front face
            w_reflection_origin = p_outside_point.transform(primitive_to_world)
            w_transmission_origin = p_inside_point.transform(primitive_to_world)

        # obtain surface space transforms
        primitive_to_surface, surface_to_primitive = _generate_surface_transforms(p_normal)
        world_to_surface = primitive_to_surface.mul(world_to_primitive)
        surface_to_world = primitive_to_world.mul(surface_to_primitive)

        # convert ray direction to surface space incident direction
        s_incoming = ray.direction.transform(world_to_surface).neg()

        # bsdf sampling
        return self.evaluate_shading(world, ray, s_incoming, w_reflection_origin, w_transmission_origin, exiting, world_to_surface, surface_to_world, intersection)

    cpdef Spectrum evaluate_shading(self, World world, Ray ray, Vector3D s_incoming,
                                    Point3D w_reflection_origin, Point3D w_transmission_origin, bint back_face,
                                    AffineMatrix3D world_to_surface, AffineMatrix3D surface_to_world,
                                    Intersection intersection):

        raise NotImplementedError("Virtual method evaluate_shading() has not been implemented.")


# Surface space
#
# to simplify maths:
# normal aligned (flipped) to sit on same side of surface as incoming ray
# incoming ray vector is aligned to point out of the surface
# surface space normal is aligned to lie along +ve Z-axis i.e. Normal3D(0, 0, 1)
#
# The w_reflection_origin and w_transmission_origin points are provided as
# ray launch points. These points are guaranteed to prevent same-surface
# re-intersections. The reflection origin lies on the same side of the
# surface as the incoming ray, the transmission origin lies on the opposite
# side of the surface.
#
# back_face is true if the ray is on the back side of the primitive surface,
# true if on the front side (ie on the side of the primitive surface normal)
cdef class ContinuousBSDF(Material):
    """
    A base class for materials implementing a continuous BSDF.
    """

    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Primitive primitive, Point3D p_hit_point,
                                    bint exiting, Point3D p_inside_point, Point3D p_outside_point,
                                    Normal3D p_normal, AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world,
                                    Intersection intersection):

        cdef:
            double pdf, pdf_importance, pdf_bsdf
            Vector3D w_outgoing, s_incoming, s_outgoing
            Point3D w_hit_point, w_reflection_origin, w_transmission_origin
            AffineMatrix3D world_to_surface, surface_to_world, primitive_to_surface, surface_to_primitive

        # surface space is aligned relative to the incoming ray
        # define ray launch points and orient normal appropriately
        if exiting:

            # ray incident on back face
            w_reflection_origin = p_inside_point.transform(primitive_to_world)
            w_transmission_origin = p_outside_point.transform(primitive_to_world)

            # flip normal
            p_normal = p_normal.neg()

        else:

            # ray incident on front face
            w_reflection_origin = p_outside_point.transform(primitive_to_world)
            w_transmission_origin = p_inside_point.transform(primitive_to_world)

        # obtain surface space transforms
        primitive_to_surface, surface_to_primitive = _generate_surface_transforms(p_normal)
        world_to_surface = primitive_to_surface.mul(world_to_primitive)
        surface_to_world = primitive_to_world.mul(surface_to_primitive)

        # convert ray direction to surface space incident direction
        s_incoming = ray.direction.transform(world_to_surface).neg()

        if ray.importance_sampling and world.has_important_primitives():

            w_hit_point = p_hit_point.transform(primitive_to_world)

            # multiple importance sampling
            if probability(ray.get_important_path_weight()):

                # sample important path pdf
                w_outgoing = world.important_direction_sample(w_hit_point)
                s_outgoing = w_outgoing.transform(world_to_surface)

            else:

                # sample bsdf pdf
                s_outgoing = self.sample(s_incoming, exiting)
                w_outgoing = s_outgoing.transform(surface_to_world)

            # compute combined pdf
            pdf_important = world.important_direction_pdf(w_hit_point, w_outgoing)
            pdf_bsdf = self.pdf(s_incoming, s_outgoing, exiting)
            pdf = ray.get_important_path_weight() * pdf_important + (1 - ray.get_important_path_weight()) * pdf_bsdf

            # evaluate bsdf and normalise
            spectrum = self.evaluate_shading(world, ray, s_incoming, s_outgoing, w_reflection_origin, w_transmission_origin, exiting, world_to_surface, surface_to_world, intersection)
            spectrum.div_scalar(pdf)
            return spectrum

        else:

            # bsdf sampling
            s_outgoing = self.sample(s_incoming, exiting)
            spectrum = self.evaluate_shading(world, ray, s_incoming, s_outgoing, w_reflection_origin, w_transmission_origin, exiting, world_to_surface, surface_to_world, intersection)
            pdf = self.pdf(s_incoming, s_outgoing, exiting)
            spectrum.div_scalar(pdf)
            return spectrum

    cpdef double pdf(self, Vector3D s_incoming, Vector3D s_outgoing, bint back_face):

        raise NotImplementedError("Virtual method pdf() has not been implemented.")

    cpdef Vector3D sample(self, Vector3D s_incoming, bint back_face):

        raise NotImplementedError("Virtual method sample() has not been implemented.")

    cpdef Spectrum evaluate_shading(self, World world, Ray ray, Vector3D s_incoming, Vector3D s_outgoing,
                                    Point3D w_reflection_origin, Point3D w_transmission_origin, bint back_face,
                                    AffineMatrix3D world_to_surface, AffineMatrix3D surface_to_world,
                                    Intersection intersection):

        raise NotImplementedError("Virtual method evaluate_shading() has not been implemented.")

    cpdef double bsdf(self, Vector3D s_incident, Vector3D s_reflected, double wavelength):
        """
        Returns the surface bi-directional scattering distribution function (BSDF).
         
        The BSDF is calculated for the given wavelength, incoming and outgoing surface space directions.
        
        :param Vector3D s_incident: The surface space incident vector, :math:`\omega_i`.
        :param Vector3D s_reflected: The surface space reflected vector, :math:`\omega_o`.
        :param float wavelength: The wavelength :math:`\lambda` at which to perform the BSDF evaluation.
        :return: The BSDF value, :math:`BSDF(\omega_i, \omega_o, \lambda)`
        """

        raise NotImplementedError("This ContinuousBSDF material has not implemented the bsdf() method.")


cdef tuple _generate_surface_transforms(Normal3D normal):
    """
    Calculates and populates the surface space transform attributes.

    :param normal: Primitive space surface normal.
    :return: Tuple containing primitive to surface and surface to primitive transforms.
    """

    cdef:
        Vector3D tangent, bitangent
        AffineMatrix3D primitive_to_surface, surface_to_primitive

    tangent = normal.orthogonal()
    bitangent = normal.cross(tangent)

    primitive_to_surface = new_affinematrix3d(
        tangent.x, tangent.y, tangent.z, 0.0,
        bitangent.x, bitangent.y, bitangent.z, 0.0,
        normal.x, normal.y, normal.z, 0.0,
        0.0, 0.0, 0.0, 1.0
    )

    surface_to_primitive = new_affinematrix3d(
        tangent.x, bitangent.x, normal.x, 0.0,
        tangent.y, bitangent.y, normal.y, 0.0,
        tangent.z, bitangent.z, normal.z, 0.0,
        0.0, 0.0, 0.0, 1.0
    )

    return primitive_to_surface, surface_to_primitive
