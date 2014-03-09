# cython: language_level=3

from math import sin, cos, tan, atan, pi
from time import time
from numpy import array, zeros
from matplotlib.pyplot import imshow, imsave, show, ion, ioff, clf, figure, draw

from raysect.core.math.point cimport Point
from raysect.core.math.vector cimport Vector
from raysect.core.math.normal cimport Normal
from raysect.core.math.affinematrix cimport AffineMatrix, translate, rotate
from raysect.core.scenegraph.world cimport World
from raysect.core.scenegraph.primitive cimport Primitive
from raysect.core.scenegraph.observer cimport Observer
from raysect.core.classes cimport Ray, Intersection, Material, SurfaceResponce, VolumeResponce
from raysect.primitives.sphere cimport Sphere

# material responces -----------------------------------------------------------

cdef class RGB:

    cdef public double r
    cdef public double g
    cdef public double b

    def __init__(self, double r = 0.0, double g = 0.0, double b = 0.0):

        self.r = r
        self.g = g
        self.b = b

    def __str__(self):

        return "RGB(" + str(self.r) + ", " + str(self.g) + ", " + str(self.b) + ")"


cdef class SurfaceRGB(SurfaceResponce):

    cdef public RGB intensity

    def __init__(self, RGB intensity = RGB(0, 0, 0)):

        self.intensity = intensity


cdef class VolumeRGB(VolumeResponce):

    cdef public RGB intensity
    cdef public RGB attenuation

    def __init__(self, RGB intensity = RGB(0, 0, 0), RGB attenuation = RGB(0, 0, 0)):

        self.intensity = intensity
        self.attenuation = attenuation


# materials --------------------------------------------------------------------

cdef class Glow(Material):

    cdef public double red
    cdef public double green
    cdef public double blue

    def __init__(self, double red = 1.0, double green = 1.0, double blue = 1.0):

        self.red = red
        self.green = green
        self.blue = blue

    cpdef SurfaceResponce evaluate_surface(self, World world, Ray ray, Primitive primitive, Point hit_point,
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

    cpdef VolumeResponce evaluate_volume(self, World world, Ray ray, Point entry_point, Point exit_point,
                                         AffineMatrix to_local, AffineMatrix to_world):

        cdef double length

        # calculate ray length
        length = entry_point.vector_to(exit_point).length

        # calculate light contribution (defined as intensity per meter)
        return VolumeRGB(RGB(self.red * length, self.green * length, self.blue * length), RGB(0, 0, 0))


cdef class Checkerboard(Material):

    cdef double _scale
    cdef double _rscale
    cdef RGB colourA
    cdef RGB colourB

    def __init__(self):

        self._scale = 1.0
        self._rscale = 1.0
        self.colourA = RGB(0.2, 0.2, 0.250)
        self.colourB = RGB(0.1, 0.1, 0.125)

    property scale:

        def __get__(self):

            return self._scale

        def __set__(self, double v):

            self._scale = v
            self._rscale = 1 / v

    cpdef SurfaceResponce evaluate_surface(self, World world, Ray ray, Primitive primitive, Point hit_point,
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

    cpdef VolumeResponce evaluate_volume(self, World world, Ray ray, Point entry_point, Point exit_point,
                                         AffineMatrix to_local, AffineMatrix to_world):

        return VolumeRGB(RGB(0, 0, 0), RGB(0, 0, 0))

    cdef inline bint _flip(self, bint v, double p):

        # round to less precision to avoid numerical precision issues (hardcoded to 1nm min scale!)
        p = round(p, 9)

        # generates check pattern from [0, inf]
        if abs(self._rscale * p) % 2 >= 1.0:

            v = not v

        # invert pattern for negative
        if p < 0:

            v = not v

        return v

# ray --------------------------------------------------------------------------

cdef class RayRGB(Ray):

    cdef public double max_depth
    cdef public double depth

    def __init__(self, Point origin = Point([0,0,0]), Vector direction = Vector([0,0,1]),
                 double min_distance = 0.0, double max_distance = float('inf'),
                 double max_depth = 15):

        super().__init__(origin, direction, min_distance, max_distance)

        self.max_depth = max_depth
        self.depth = 0

    cpdef RGB trace(self, World world):

        cdef RGB spectrum
        cdef Intersection intersection
        cdef Material material
        cdef SurfaceRGB sresponce
        cdef VolumeRGB vresponce
        cdef list primitives
        cdef Primitive primitive
        cdef Point entry_point, exit_point

        spectrum = RGB(0, 0, 0)

        if self.depth >= self.max_depth:

            return spectrum

        intersection = world.hit(self)

        if intersection is not None:

            # surface contribution
            material = intersection.primitive.material
            sresponce = <SurfaceRGB> material.evaluate_surface(world, self,
                                                 intersection.primitive,
                                                 intersection.hit_point,
                                                 intersection.exiting,
                                                 intersection.inside_point,
                                                 intersection.outside_point,
                                                 intersection.normal,
                                                 intersection.to_local,
                                                 intersection.to_world)

            spectrum.r = sresponce.intensity.r
            spectrum.g = sresponce.intensity.g
            spectrum.b = sresponce.intensity.b

            # volume contribution - TODO: deal with min_distance and max_distance if no intersection occurs.
            primitives = world.inside(self.origin)

            entry_point = self.origin
            exit_point = intersection.hit_point.transform(intersection.to_world)

            for primitive in primitives:

                vresponce = <VolumeRGB> primitive.material.evaluate_volume(world,
                                                              self,
                                                              entry_point,
                                                              exit_point,
                                                              primitive.to_root(),
                                                              primitive.to_local())

                spectrum.r = (1 - vresponce.attenuation.r) * spectrum.r + vresponce.intensity.r
                spectrum.g = (1 - vresponce.attenuation.g) * spectrum.g + vresponce.intensity.g
                spectrum.b = (1 - vresponce.attenuation.b) * spectrum.b + vresponce.intensity.b

        return spectrum

    cpdef Ray spawn_daughter(self, Point origin, Vector direction):

        cdef RayRGB ray

        ray = RayRGB(origin, direction)

        ray.max_depth = self.max_depth
        ray.depth = self.depth + 1

        return ray

# observer ---------------------------------------------------------------------

cdef class PinholeCamera(Observer):

    cdef tuple _pixels
    cdef double _fov
    cdef object frame
    cdef bint display_progress
    cdef double display_update_time

    def __init__(self, pixels = (640, 480), fov = 40, parent = None, transform = AffineMatrix(), name = ""):

        super().__init__(parent, transform, name)

        self.pixels = pixels
        self.fov = fov
        self.frame = None

        self.display_progress = True
        self.display_update_time = 2.0

    property pixels:

        def __get__(self):

            return self._pixels

        def __set__(self, pixels):

            if len(pixels) != 2:

                raise ValueError("Pixel dimensions of camera framebuffer must be a tuple containing the x and y pixel counts.")

            self._pixels = pixels

    property fov:

        def __get__(self):

            return self._fov

        def __set__(self, fov):

            if fov <= 0:

                raise ValueError("Field of view angle can not be less than or equal to 0 degrees.")

            self._fov = fov

    cpdef observe(self):

        self.frame = zeros((self._pixels[1], self._pixels[0], 3))

        if isinstance(self.root, World) == False:

            raise TypeError("Observer is not conected to a scenegraph containing a World object.")

        world = self.root

        max_pixels = max(self._pixels)

        if max_pixels > 0:

            # max width of image plane at 1 meter
            image_max_width = 2 * tan(pi / 180 * 0.5 * self._fov)

            # pixel step and start point in image plane
            image_delta = image_max_width / (max_pixels - 1)

            # start point of scan in image plane
            image_start_x = 0.5 * self._pixels[0] * image_delta
            image_start_y = 0.5 * self._pixels[1] * image_delta

        else:

            # single ray on axis
            image_delta = 0
            image_start_x = 0
            image_start_y = 0

        ray = RayRGB()

        if self.display_progress:

            self.display()
            display_timer = time()

        for y in range(0, self._pixels[1]):

            print(y)

            for x in range(0, self._pixels[0]):

                # ray angles
                theta = atan(image_start_x - image_delta * x)
                phi = atan(image_start_y - image_delta * y)

                # calculate ray parameters
                origin = Point([0, 0, 0])
                direction = Vector([sin(theta),
                                    cos(theta) * sin(phi),
                                    cos(theta) * cos(phi)])

                # convert to world space
                ray.origin = origin.transform(self.to_root())
                ray.direction = direction.transform(self.to_root())

                # trace and accumulate
                responce = ray.trace(world)
                self.frame[y, x, 0] = responce.r
                self.frame[y, x, 1] = responce.g
                self.frame[y, x, 2] = responce.b

                if self.display_progress and (time() - display_timer) > self.display_update_time:

                    self.display()
                    display_timer = time()

        if self.display_progress:

            self.display()

    def display(self, whitepoint = 1.0, gamma = 2.2):

        image = self.frame.copy()
        image = self._adjust(image, whitepoint, gamma)
        clf()
        imshow(image, aspect = "equal", origin = "upper")
        draw()

    def _adjust(self, image, whitepoint, gamma):

        # adjust whitepoint and clamp image to [0, 1.0]
        image[image > whitepoint] = whitepoint
        image = image / float(whitepoint)

        # apply gamma correction
        image = image**(1.0 / float(gamma))

        return image

# demo script ------------------------------------------------------------------

def demo():

    print("Raysect test script")

    world = World()
    sphere_r = Sphere(2.0, world, translate(0, 1, 5), Glow(0.25, 0, 0))
    sphere_g = Sphere(2.0, world, translate(0.707, -0.707, 5), Glow(0, 0.25, 0))
    sphere_b = Sphere(2.0, world, translate(-0.707, -0.707, 5), Glow(0, 0, 0.25))
    sphere_world = Sphere(100.0, world, rotate(0, 0, 0), Checkerboard())
    sphere_world.material.scale = 5

    observer = PinholeCamera((640, 480), 45, world, translate(0, 0, -5) * rotate(0, 0, 0))

    ion()
    figure(1)

    t = time()
    observer.observe()
    t = time() - t
    print("Render time was " + str(t) + " seconds")

    ioff()
    observer.display()
    show()


demo()


