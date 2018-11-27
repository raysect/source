
Glossary
========

A glossary of the key concepts used in Raysect.

Affine Matrix
   A 4x4 matrix which defines a transformation between two coordinate
   systems. See :class:`.AffineMatrix3D`.

Flux
   See radiant flux.

Intensity
   The radiant flux per solid angle. Note that this is not the same
   as radiance, and in practice not very useful except for comparing
   point sources.

Intersection
   A ray-primitive intersection that describes the point of intersection
   between the given ray object and a piece of geometry (a primitive) in
   the scene :class:`.Intersection`.

Irradiance
   The area density of radiant flux passing through a surface, symbol
   :math:`E`, with units W/m^2.

Material
   A material object provides an associated primitive with physical
   surface and volume properties. e.g. :class:`.core.material.Material`
   and :class:`.optical.material.Material`

Node
   A fundamental element of the scene-graph with its own local coordinate
   system :class:`.Node`.

Observer
   An observing node of the the scene-graph that generates rays to sample
   the scene. Observers can be 0D (Pixel, FibreOptic) or 2D
   (PinholeCamera). Observers are responsible for all the geometric
   effects of the observing process, such as the pixel area and
   solid angle sensitivity that define an effective etendue for the
   observer. After ray-tracing the results of all the samples are
   passed to a list of connected pipeline objects for further data
   procressing. See :ref:`observers-page` for more examples.

Pipeline
   Pipelines are responsible for the data analysis of observers after
   ray-tracing. They process the mean spectral radiance at each sample
   location (could be 0D or 2D) into the more specific data type the
   user is interested it. For example, users may be interested in
   converting mean spectral radiance values to the more human friendly
   RGB colour or raw power values. See :ref:`observers-page` for more
   examples.

Primitive
   A geometric structure that interacts with rays and is also a scene-graph
   node. May define a closed or open surface, and a volume. Examples of
   primitives are geometric shapes, such as the Box and Cylinder, as well
   as structured meshes. All primitives have a material applied to their
   surface and volume. See the :ref:`primitives-module`.

Radiance
   Radiance is the flux density per unit area per steradian, W/m^2/str, given
   the symbol :math:`L`. Radiance is the most fundamental radiometric quantity
   since all other quantities can be calculated as integrals of radiance over
   areas and solid angles. Radiance is the natural quantity for ray-tracing
   because it remains constant along rays in empty space.

Radiant Flux
   Also known as power, the total amount of energy passing through a surface
   in one second, i.e. units of watts (W) with the symbol (:math:`\Phi`).

Ray
   Describes a line in space with an origin and direction. Ray's are used to
   trace() the world (e.g. :class:`.core.ray.Ray` and :class:`.optical.ray.Ray`).

Scene
   A particular scene-graph instance consisting of geometry and cameras that
   correspond to a physical scene.

Scenegraph
   A tree structure that represents a nested set of coordinate systems. Changes
   to the coordinate system at any node level are cascaded to all children. For
   example, suppose you have a car node that contains separate child nodes for
   each of the wheels and car body. Any transform applied to the car node,
   would also be applied to each of the cars component parts. The scenegraph
   makes it easy to manage complex scenes by allowing complex nodes to have
   their own local coordinate system. Any instanced node objects (e.g. boxes,
   lenses, etc) that are not connected to the scene-graph will not be included
   in the ray-tracing. It is possible to have multiple worlds (scene-graphs) in
   the same script for cases where you need to trace two different scenes and
   compare the results.

Solid Angle
   The two dimensional angle in three dimensional space that an object subtends
   at a point. It represents how large an object appears to an observer at that
   point. Measured in units of steradian (str). There are :math:`4\pi` steradians
   in a sphere.

Transform
   A 4x4 affine matrix which defines a transformation between two coordinate
   systems. See :class:`.AffineMatrix3D`.

World
   The root node of the scene-graph on which ray-tracing is performed, it also
   holds the acceleration structures used by the ray-tracer. See
   :class:`.core.scenegraph.world.World` and :class:`.optical.scenegraph.world.World`.
