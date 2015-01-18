import numpy as np
from matplotlib.pyplot import imshow, imsave, show, ion, ioff, clf, figure, draw, pause
from multiprocessing import Pool, Queue, Pipe, Process, cpu_count
from time import time

from raysect.optical.colour import resample_ciexyz, spectrum_to_ciexyz, ciexyz_to_srgb
from raysect.optical import World, translate, rotate, Point, Vector, Ray, d65_white, ConstantSF, SampledSF
from raysect.optical.material.emitter import UniformVolumeEmitter, UniformSurfaceEmitter, Checkerboard
from raysect.primitive import Sphere, Box, Cylinder, Union, Intersect, Subtract
from raysect.optical.material.glass_libraries import schott

# kludge to fix matplotlib 1.4 ion() idiocy
import sys
sys.ps1 = 'SOMETHING'


def display(rgb_frame):

    clf()
    imshow(rgb_frame, aspect="equal", origin="upper")
    draw()
    show()

    # workaround for interactivity for QT backend
    pause(0.1)


def producer(pixels, image_start_x, image_start_y, image_delta,
             min_wavelength, max_wavelength, spectral_samples, max_depth,
             task_queue):

    for y in range(0, pixels[1]):

        # submit tasks for current line
        for x in range(0, pixels[0]):

            # calculate ray parameters
            origin = Point(0, 0, -4)
            direction = Vector(image_start_x - image_delta * x,
                               image_start_y - image_delta * y,
                               1.0).normalise()

            # build task
            ray = Ray(origin, direction, min_wavelength=min_wavelength, max_wavelength=max_wavelength, num_samples=spectral_samples, max_depth=max_depth)
            task = ((x, y), ray)

            # submit task
            task_queue.put(task)


def worker(pid, world, task_queue, result_queue):

    while True:

        # request next task
        task = task_queue.get()

        # have we been commanded to shutdown?
        if task == "STOP":
            break

        # decode task
        (id, ray) = task

        # trace
        spectrum = ray.trace(world)

        # encode result and send
        result = (id, spectrum)
        result_queue.put(result)


# setup render
fov = 60
pixels = (512, 512)
min_wavelength, max_wavelength, spectral_samples = 375.0, 740.0, 20
max_depth = 20
display_update_time = 10
num_processes = cpu_count()
ion()

# world
world = World()
cyl_x = Cylinder(1, 4.2, transform=rotate(90, 0, 0)*translate(0, 0, -2.1))
cyl_y = Cylinder(1, 4.2, transform=rotate(0, 90, 0)*translate(0, 0, -2.1))
cyl_z = Cylinder(1, 4.2, transform=rotate(0, 0, 0)*translate(0, 0, -2.1))
cube = Box(Point(-1.5, -1.5, -1.5), Point(1.5, 1.5, 1.5))
sphere = Sphere(2.0)

# Intersect(sphere, cube, world, translate(-2.1,2.1,2.5)*rotate(30, -20, 0), schott("N-LAK9"))
# Intersect(sphere, cube, world, translate(2.1,2.1,2.5)*rotate(-30, -20, 0), schott("SF6"))
# Intersect(sphere, cube, world, translate(2.1,-2.1,2.5)*rotate(-30, 20, 0), schott("LF5G19"))
# Intersect(sphere, cube, world, translate(-2.1,-2.1,2.5)*rotate(30, 20, 0), schott("N-BK7"))

# Sphere(2.0, world, translate(-2.1,2.1,2.5)*rotate(30, -20, 0), schott("N-LAK9"))
# Sphere(2.0, world, translate(2.1,2.1,2.5)*rotate(-30, -20, 0), schott("SF6"))
# Sphere(2.0, world, translate(2.1,-2.1,2.5)*rotate(-30, 20, 0), schott("LF5G19"))
# Sphere(2.0, world, translate(-2.1,-2.1,2.5)*rotate(30, 20, 0), schott("N-BK7"))

Intersect(sphere, Subtract(cube, Union(Union(cyl_x, cyl_y), cyl_z)), world, translate(-2.1,2.1,2.5)*rotate(30, -20, 0), schott("N-LAK9"))
Intersect(sphere, Subtract(cube, Union(Union(cyl_x, cyl_y), cyl_z)), world, translate(2.1,2.1,2.5)*rotate(-30, -20, 0), schott("SF6"))
Intersect(sphere, Subtract(cube, Union(Union(cyl_x, cyl_y), cyl_z)), world, translate(2.1,-2.1,2.5)*rotate(-30, 20, 0), schott("LF5G19"))
Intersect(sphere, Subtract(cube, Union(Union(cyl_x, cyl_y), cyl_z)), world, translate(-2.1,-2.1,2.5)*rotate(30, 20, 0), schott("N-BK7"))

Box(Point(-50, -50, 50), Point(50, 50, 50.1), world, material=Checkerboard(4, d65_white, d65_white, 0.4, 0.8))
Box(Point(-100, -100, -100), Point(100, 100, 100), world, material=UniformSurfaceEmitter(d65_white, 0.1))

# setup queues
task_queue = Queue()
result_queue = Queue()

# start worker processes
processes = []
for pid in range(num_processes):
    process = Process(target=worker, args=(pid,world, task_queue, result_queue))
    process.start()
    processes.append(process)

# create frames
xyz_frame = np.zeros((pixels[1], pixels[0], 3))
rgb_frame = np.zeros((pixels[1], pixels[0], 3))

# max width of image plane at 1 meter
image_max_width = 2 * np.tan(np.pi / 180 * 0.5 * fov)

# pixel step and start point in image plane
max_pixels = max(pixels)
image_delta = image_max_width / (max_pixels - 1)

# start point of scan in image plane
image_start_x = 0.5 * pixels[0] * image_delta
image_start_y = 0.5 * pixels[1] * image_delta

resampled_xyz = resample_ciexyz(min_wavelength,
                                max_wavelength,
                                spectral_samples)

# initialise statistics
display(rgb_frame)
display_timer = time()
start_time = time()

# start task producer
Process(target=producer,
        args=(pixels, image_start_x, image_start_y, image_delta,
            min_wavelength, max_wavelength, spectral_samples, max_depth,
            task_queue)).start()

# collect results
for i in range(0, pixels[0]*pixels[1]):

    # obtain result
    location, spectrum = result_queue.get()
    fx, fy = location

    # convert spectrum to CIE XYZ and accumulate
    xyz = spectrum_to_ciexyz(spectrum, resampled_xyz)
    xyz_frame[fy, fx, 0] += xyz[0]
    xyz_frame[fy, fx, 1] += xyz[1]
    xyz_frame[fy, fx, 2] += xyz[2]

    # update display image
    rgb_frame[fy, fx, :] = ciexyz_to_srgb(*xyz_frame[fy, fx, :])

    if (time() - display_timer) > display_update_time:

        print("{:0.2f}%: pixel {}/{}".format(100 * i / (pixels[0]*pixels[1]), i, pixels[0]*pixels[1]))
        #print("Refreshing display...")
        display(rgb_frame)
        display_timer = time()

# close statistics
elapsed_time = time() - start_time
print("Render complete - time elapsed {:0.3f}s".format(elapsed_time))

display(rgb_frame)

# stop workers
for process in processes:
    task_queue.put("STOP")


ioff()
display(rgb_frame)
show()