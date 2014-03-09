from math import sin, cos, tan, atan, pi
from time import time
from numpy import array, zeros
from matplotlib.pyplot import imshow, imsave, show, ion, ioff, clf, figure, draw
from raysect.core import Ray, World, Observer, Point, Vector, AffineMatrix, translate, rotate
from raysect.core import Material, SurfaceResponce, VolumeResponce
from raysect.primitives import Sphere

# material responces -----------------------------------------------------------

class RGB:

    def __init__(self, r = 0.0, g = 0.0, b = 0.0):

        self.r = r
        self.g = g
        self.b = b

    def __str__(self):

        return "RGB(" + str(self.r) + ", " + str(self.g) + ", " + str(self.b) + ")"


class SurfaceRGB(SurfaceResponce):

    def __init__(self, intensity = RGB(0, 0, 0)):

        self.intensity = intensity


class VolumeRGB(VolumeResponce):

    def __init__(self, intensity = RGB(0, 0, 0), attenuation = RGB(0, 0, 0)):

        self.intensity = intensity
        self.attenuation = attenuation


# materials --------------------------------------------------------------------

class Glow(Material):

    def __init__(self, red = 1.0, green = 1.0, blue = 1.0):

        self.red = red
        self.green = green
        self.blue = blue

    def evaluate_surface(self, world, ray, primitive, hit_point, exiting, inside_point, outside_point, normal, to_local, to_world):

        # are we entering or leaving surface?
        if exiting:

            # ray leaving surface, use outer point for ray origin
            origin = outside_point.transform(to_world)

        else:

            # ray entering surface, use inner point for ray origin
            origin = inside_point.transform(to_world)

        daughter_ray = ray.spawn_daughter(origin, ray.direction)

        return SurfaceRGB(daughter_ray.trace(world))

    def evaluate_volume(self, world, ray, entry_point, exit_point, to_local, to_world):

        # calculate ray length
        length = entry_point.vector_to(exit_point).length

        # calculate light contribution (defined as intensity per meter)
        return VolumeRGB(RGB(self.red * length, self.green * length, self.blue * length), RGB(0, 0, 0))


class Checkerboard(Material):

    def __init__(self):

        self._scale = 1.0
        self._rscale = 1.0
        self.colourA = RGB(0.2, 0.2, 0.250)
        self.colourB = RGB(0.1, 0.1, 0.125)

    @property
    def scale(self):

        return self._scale

    @scale.setter
    def scale(self, v):

        self._scale = v
        self._rscale = 1 / v

    def evaluate_surface(self, world, ray, primitive, hit_point, exiting, inside_point, outside_point, normal, to_local, to_world):

        v = False

        # generate check pattern
        v = self._flip(v, hit_point.x)
        v = self._flip(v, hit_point.y)
        v = self._flip(v, hit_point.z)

        # apply "colours"
        return SurfaceRGB(RGB(self.colourA.r * v + self.colourB.r * (not v),
                              self.colourA.g * v + self.colourB.g * (not v),
                              self.colourA.b * v + self.colourB.b * (not v)))

    def evaluate_volume(self, world, ray, entry_point, exit_point, to_local, to_world):

        return VolumeRGB(RGB(0, 0, 0), RGB(0, 0, 0))

    def _flip(self, v, p):

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

class RayRGB(Ray):

    def __init__(self, origin = Point([0,0,0]), direction = Vector([0,0,1]), min_distance = 0.0, max_distance = float('inf'),
                 max_depth = 15):

        super().__init__(origin, direction, min_distance, max_distance)

        self.max_depth = max_depth
        self.depth = 0

    def trace(self, world):

        spectrum = RGB(0, 0, 0)

        if self.depth >= self.max_depth:

            return spectrum

        intersection = world.hit(self)

        if intersection is not None:

            # surface contribution
            material = intersection.primitive.material
            responce = material.evaluate_surface(world, self,
                                                 intersection.primitive,
                                                 intersection.hit_point,
                                                 intersection.exiting,
                                                 intersection.inside_point,
                                                 intersection.outside_point,
                                                 intersection.normal,
                                                 intersection.to_local,
                                                 intersection.to_world)

            spectrum.r = responce.intensity.r
            spectrum.g = responce.intensity.g
            spectrum.b = responce.intensity.b

            # volume contribution - TODO: deal with min_distance and max_distance if no intersection occurs.
            primitives = world.inside(self.origin)

            entry_point = self.origin
            exit_point = intersection.hit_point.transform(intersection.to_world)

            for primitive in primitives:

                responce = primitive.material.evaluate_volume(world,
                                                              self,
                                                              entry_point,
                                                              exit_point,
                                                              primitive.to_root(),
                                                              primitive.to_local())

                spectrum.r = (1 - responce.attenuation.r) * spectrum.r + responce.intensity.r
                spectrum.g = (1 - responce.attenuation.g) * spectrum.g + responce.intensity.g
                spectrum.b = (1 - responce.attenuation.b) * spectrum.b + responce.intensity.b

        return spectrum

    def spawn_daughter(self, origin, direction):

        ray = RayRGB(origin, direction)

        ray.max_depth = self.max_depth
        ray.depth = self.depth + 1

        return ray

# observer ---------------------------------------------------------------------

class PinholeCamera(Observer):

    def __init__(self, pixels = (640, 480), fov = 40, parent = None, transform = AffineMatrix(), name = ""):

        super().__init__(parent, transform, name)

        self.pixels = pixels
        self.fov = fov
        self.frame = None

        self.display_progress = True
        self.display_update_time = 2.0

    @property
    def pixels(self):

        return self._pixels

    @pixels.setter
    def pixels(self, pixels):

        if len(pixels) != 2:

            raise ValueError("Pixel dimensions of camera framebuffer must be a tuple containing the x and y pixel counts.")

        self._pixels = pixels

    @property
    def fov(self):

        return self._fov

    @fov.setter
    def fov(self, fov):

        if fov <= 0:

            raise ValueError("Field of view angle can not be less than or equal to 0 degrees.")

        self._fov = fov

    def observe(self):

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

