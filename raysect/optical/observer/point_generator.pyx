

from raysect.core.math.point cimport Point3D, Point2D
from raysect.core.math.random import point_disk, uniform
from raysect.core import AffineMatrix3D


cdef class PointGenerator:
    """
    Base class for defining the sampling area for rays launched by an observer. Observers use the
    VectorGenerator and PointGenerator classes to build N rays for sampling.
    """

    def __init__(self, transform=AffineMatrix3D()):
        self.transform = transform

    def __call__(self, n):
        """
        :param int n: Generate n points that sample this observers surface area.
        """
        return self.sample(n)

    cpdef list sample(self, int n):
        """
        :param int n: Generate n vectors that sample this observers acceptance solid angle.
        """
        raise NotImplemented("The method sample(n) for this point generator needs to be implemented.")


cdef class SinglePointGenerator(PointGenerator):

    cpdef list sample(self, int n):
        cdef list results
        cdef int i

        results = []
        for i in range(n):
            results.append(Point3D(0, 0, 0).transform(self.transform))
        return results


cdef class CircularPointGenerator(PointGenerator):

    def __init__(self, radius, transform=AffineMatrix3D()):
        """
        :param double radius: The radius of the circular sampling area of this observer.
        """
        super().__init__(transform=transform)
        self.radius = radius

    cpdef list sample(self, int n):
        cdef list results
        cdef int i
        cdef double radius = self.radius
        cdef Point2D random_point

        results = []
        for i in range(n):
            random_point = point_disk()
            results.append(Point3D(random_point.x * radius, random_point.y * radius, 0).transform(self.transform))
        return results


cdef class RectangularPointGenerator(PointGenerator):

    def __init__(self, width, height, transform=AffineMatrix3D()):
        """
        :param double width: The width of the rectangular sampling area of this observer.
        :param double height: The height of the rectangular sampling area of this observer.
        """
        super().__init__(transform=transform)
        self.width = width
        self.height = height

    cpdef list sample(self, int n):
        cdef list results
        cdef int i
        cdef double u, v, width_offset = self.width/2, height_offset = self.height/2

        results = []
        for i in range(n):
            u = uniform()*self.width - width_offset
            v = uniform()*self.height - height_offset
            results.append(Point3D(u, v, 0).transform(self.transform))
        return results
