
from raysect.core import Primitive, AffineMatrix, Normal, Point, Vector
from raysect.core.classes import Material, Intersection
from raysect.core.acceleration.boundingbox import BoundingBox
from scipy import sqrt, fabs

# cython doesn't have a built-in infinity constant, this compiles to +infinity
INFINITY = 1e999

# bounding box is padded by a small amount to avoid numerical accuracy issues
BOX_PADDING = 1e-9

# additional ray distance to avoid re-hitting the same surface point
EPSILON = 1e-9

# object type enumeration
NO_TYPE = -1
CONE = 0
SLAB = 1

# slab face enumeration
NO_FACE = -1
LOWER_FACE = 0


class Cone(Primitive):

    def __init__(self, radius=0.5, height=1.0, parent=None, transform=AffineMatrix(), material=Material(), name=""):

        super().__init__(parent, transform, material, name)

        # validate radius and height values
        if radius <= 0.0:
            raise ValueError("Cone radius cannot be less than or equal to zero.")
        if height <= 0.0:
            raise ValueError("Cone height cannot be less than or equal to zero.")
        self._radius = radius
        self._height = height

        # Only needed for next CSG intersection (i.e. when you have overlapping primitives)
        # initialise next intersection caching and control attributes
        # self._further_intersection = False
        # self._next_t = 0.0
        # self._cached_origin = None
        # self._cached_direction = None
        # self._cached_ray = None
        # self._cached_face = NO_FACE
        # self._cached_type = NO_TYPE

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value <= 0.0:
            raise ValueError("Cone radius cannot be less than or equal to zero.")
        self._radius = value

        # the next intersection cache has been invalidated by the geometry change
        # self._further_intersection = False

        # any geometry caching in the root node is now invalid, inform root
        self.notify_root()

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        if value <= 0.0:
            raise ValueError("Cone height cannot be less than or equal to zero.")
        self._height = value

        # the next intersection cache has been invalidated by the geometry change
        # self._further_intersection = False
        # any geometry caching in the root node is now invalid, inform root
        self.notify_root()

    def __str__(self):
        """String representation."""
        if self.name == "":
            return "<Cone at " + str(hex(id(self))) + ">"
        else:
            return self.name + " <Cone at " + str(hex(id(self))) + ">"

    def hit(self, ray):

        # reset the next intersection cache
        # self._further_intersection = False

        # convert ray origin and direction to local space
        origin = ray.origin.transform(self.to_local())
        direction = ray.direction.transform(self.to_local())

        radius = self._radius
        height = self._height

        # Compute quadratic cone coefficients
        # based on "Physically Based Rendering - 2nd Edition", Elsevier 2010
        k = radius / height
        k = k * k
        a = direction.x * direction.x + direction.y * direction.y - k * direction.z * direction.z
        b = 2 * (direction.x * origin.x + direction.y * origin.y - k * direction.z * (origin.z-height) )
        c = origin.x * origin.x + origin.y * origin.y - k * (origin.z - height) * (origin.z - height)
        d = b * b - 4 * a * c

        # ray misses cone if there are no real roots of the quadratic
        if d < 0:
            return None

        d = sqrt(d)

        # calculate intersections
        temp = 1 / (2.0 * a)
        t0 = -(d + b) * temp
        t1 = (d - b) * temp

        # ensure t0 is always smaller than t1
        if t0 > t1:
            temp = t0
            t0 = t1
            t1 = temp

        near_point = ray.get_point_at_distance(t0).transform(self.to_local())
        far_point = ray.get_point_at_distance(t1).transform(self.to_local())

        # Are both intersections inside the cone height
        if 0 < near_point.z < height and 0 < far_point.z < height:

            # Both intersections are with the cone body
            near_intersection = t0
            near_type = CONE
            near_face = NO_FACE

            far_intersection = t1
            far_type = CONE
            far_face = NO_FACE

        # Is only one of the intersections inside the cone body, therefore other is with flat cone base.
        elif 0 < near_point.z < height or 0 < far_point.z < height:

            if near_point.z < 0 or far_point.z > height:
                # Near intersection is cone base, far is cone face
                near_intersection = -origin.z / direction.z
                near_type = SLAB
                near_face = LOWER_FACE

                far_intersection = t0
                far_type = CONE
                far_face = NO_FACE

            else:
                # Otherwise far intersection is cone bace, near is cone face.
                near_intersection = t0
                near_type = CONE
                near_face = NO_FACE

                far_intersection = (self._height - origin.z) / direction.z
                far_type = SLAB
                far_face = LOWER_FACE

        # Both intersections are outside the bounding box.
        else:
            return None

        return self._generate_intersection(ray, origin, direction, near_intersection, near_face, near_type)

    # Only used by CSG
    # cpdef Intersection next_intersection(self):
    #
    #     if not self._further_intersection:
    #         return None
    #
    #     # this is the 2nd and therefore last intersection
    #     self._further_intersection = False
    #
    #     return self._generate_intersection(self._cached_ray, self._cached_origin, self._cached_direction, self._next_t, self._cached_face, self._cached_type)

    # This function is called twice. Used in hit() and next_intersection()
    def _generate_intersection(self, ray, origin, direction, ray_distance, face, type):

        # point of surface intersection in local space
        hit_point = Point(origin.x + ray_distance * direction.x,
                          origin.y + ray_distance * direction.y,
                          origin.z + ray_distance * direction.z)

        # if hit point equals tip, set normal to up.
        if type == CONE and (hit_point.x == 0.0) and (hit_point.y == 0.0) and (hit_point.z == self.height):
            normal = Normal(0, 0, 1)

        # calculate surface normal in local space
        elif type == CONE:
            # Unit vector that points from origin to hit_point in x-y plane at the base of the cone.
            op = Normal(hit_point.x, hit_point.y, 0)
            op = op.normalise()
            heighttoradius = self.height/self.radius
            normal = Normal(op.x * heighttoradius, op.y * heighttoradius, 1/heighttoradius)
            normal = normal.normalise()

        else:
            normal = Normal(0, 0, -1)

        # displace hit_point away from surface to generate inner and outer points
        interior_offset = self._interior_offset(hit_point, normal, type)

        # TODO - make sure this is working correctly for special cases, i.e. tip.
        inside_point = Point(hit_point.x + interior_offset.x,
                             hit_point.y + interior_offset.y,
                             hit_point.z + interior_offset.z)

        outside_point = Point(hit_point.x + EPSILON * normal.x,
                              hit_point.y + EPSILON * normal.y,
                              hit_point.z + EPSILON * normal.z)

        # is ray exiting surface
        if direction.dot(normal) >= 0.0:
            exiting = True
        else:
            exiting = False

        return Intersection(ray, ray_distance, self, hit_point, inside_point, outside_point,
                            normal, exiting, self.to_local(), self.to_root())

    def _interior_offset(self, hit_point, normal, type):

        # shift away from cone surface
        if type == CONE:
            x = -EPSILON * normal.x
            y = -EPSILON * normal.y
            z = -EPSILON * normal.z
        else:
            x = 0
            y = 0

            if hit_point.x != 0.0 and hit_point.y != 0.0:

                length = sqrt(hit_point.x * hit_point.x + hit_point.y * hit_point.y)

                if (length - self._radius) < EPSILON:

                    length = 1.0 / length
                    x = -EPSILON * length * hit_point.x
                    y = -EPSILON * length * hit_point.y

        # shift away from bottom surface
        if fabs(hit_point.z) < EPSILON:
            z = EPSILON
        else:
            z = 0

        return Vector(x, y, z)

    def contains(self, point):

        # convert point to local object space
        point = point.transform(self.to_local())

        # Calculate points' distance along z axis from cone tip
        cone_dist = self.height - point.z

        # reject points that are outside the cone's height (i.e. above the cones' tip or below its base)
        if not 0 <= cone_dist <= self.height:
            return False

        # Calculate the cone radius at that point along the height axis:
        cone_radius = (cone_dist / self.height) * self.radius

        # Calculate the point's orthogonal distance from the axis to compare against the cone radius:
        orth_distance = sqrt(point.x**2 + point.y**2)

        # Points distance from axis must be less than cone radius at that height
        return orth_distance < cone_radius

    def bounding_box(self):

        box = BoundingBox()

        # calculate local bounds
        box.lower = Point(-self._radius, -self._radius, 0.0)
        box.upper = Point(self._radius, self._radius, self._height)

        # obtain local space vertices
        points = box.vertices()

        # convert points to world space and build an enclosing world space bounding box
        # a small degree of padding is added to avoid potential numerical accuracy issues
        box = BoundingBox()
        for point in points:

            box.extend(point.transform(self.to_root()), BOX_PADDING)

        return box

