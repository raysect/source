"""Compare one spectral CustomBRDF sphere at three wavelengths.

Run from the repository root:
    python demos/materials/custom_brdf.py --show

A single four-dimensional BRDF material is shared by all three renders. The
scene, roughness and monochromatic unit-radiance illuminants are identical;
only the camera's narrow wavelength band changes. Wavelengths use nm, surface
roughness uses µm. Each 0.5-degree wavelength slice is about 180 MiB, and the
material copies its input, so the default three slices need over 1 GiB during
construction. Images use one common linear radiance scale.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from raysect.core import Point3D
from raysect.core.workflow import SerialEngine
from raysect.optical import ConstantSF, World, rotate, translate
from raysect.optical.material import CustomBRDF, Lambert, UniformSurfaceEmitter
from raysect.optical.observer import PinholeCamera, RadiancePipeline2D
from raysect.primitive import Box, Sphere

RHO_D, RHO_S = 0.05, 0.75


def optical_roughness_brdf(theta_i, theta_o, delta_phi, wavelength, roughness, rho_d, rho_s):
    """Evaluate the diffuse-plus-specular optical-roughness BRDF (sr^-1).

    Implements equation (14) of P. Su, et. al., "Optical
    roughness BRDF model for reverse Monte Carlo simulation of real material
    thermal radiation transfer," Applied Optics 53, 2324-2330 (2014),
    https://doi.org/10.1364/AO.53.002324.

    theta_i and theta_o are incident and outgoing polar angles in radians,
    with both direction vectors pointing away from the surface; delta_phi is
    their relative azimuth in radians. roughness is the paper's surface
    roughness sigma-prime, wavelength is lambda, and their ratio sigma-prime /
    lambda is dimensionless: both must use the same length unit. rho_d and
    rho_s are the diffuse and specular reflectance parameters, respectively.

    At 90 degrees, the polar angle is clamped to 89.99 degrees as a numerical
    approximation. The exact double-grazing limit can depend on direction and
    is undefined; results near grazing angles require particular care.
    """
    if wavelength <= 0 or roughness <= 0:
        raise ValueError('wavelength and roughness must be positive')
    ti = np.minimum(theta_i, np.pi / 2 - np.deg2rad(0.01))
    to = np.minimum(theta_o, np.pi / 2 - np.deg2rad(0.01))
    ci, co = np.cos(ti), np.cos(to)
    dot = ci * co + np.sin(ti) * np.sin(to) * np.cos(delta_phi)
    h2 = np.maximum(0.0, (1 + dot) / 2)
    cos_h2 = np.minimum(1.0, (ci + co)**2 / np.maximum(4 * h2, 1e-300))
    tan_h2 = np.maximum(0.0, 1 - cos_h2) / np.maximum(cos_h2, 1e-300)
    alpha = roughness / wavelength
    return rho_d / np.pi + rho_s / (np.pi * alpha**2) * np.exp(-tan_h2 / alpha**2) * 2 * (1 + dot) / (ci + co)**4


def make_material(wavelengths, roughness):
    """Tabulate the model at 0.5° per angular axis, for every wavelength."""
    theta = np.linspace(0, np.pi / 2, 181)
    phi = np.linspace(0, 2 * np.pi, 720, endpoint=False)[:, None]
    table = np.empty((len(wavelengths), 181, 720, 181))
    for w, wavelength in enumerate(wavelengths):
        print(f'Tabulating {wavelength:g} nm...', flush=True)
        for i, incoming in enumerate(theta):
            table[w, i] = optical_roughness_brdf(incoming, theta[None, :], phi,
                                                 wavelength / 1000, roughness, RHO_D, RHO_S)
    return CustomBRDF(table, wavelengths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wavelengths', nargs=3, type=float, default=(700, 1000, 1300),
                        metavar=('LEFT', 'CENTER', 'RIGHT'), help='three increasing wavelengths in nm')
    parser.add_argument('--roughness', type=float, default=0.3, help='surface roughness in µm (default 0.3)')
    parser.add_argument('--width', type=int, default=180, help='width of each render (default 180)')
    parser.add_argument('--height', type=int, default=180, help='height of each render (default 180)')
    parser.add_argument('--samples', type=int, default=1000, help='samples per pixel (default 1000)')
    parser.add_argument('--output', type=Path, default=Path('custom_brdf_wavelengths.png'), help='output PNG')
    parser.add_argument('--show', action='store_true', help='also display the image')
    args = parser.parse_args()
    waves = np.asarray(args.wavelengths)
    if min(args.width, args.height, args.samples) < 1:
        parser.error('width, height and samples must be positive')
    if not np.isfinite(args.roughness) or args.roughness <= 0:
        parser.error('roughness must be finite and positive')
    if not np.all(np.isfinite(waves)) or np.any(waves <= 1) or np.any(np.diff(waves) <= 0):
        parser.error('wavelengths must be finite, strictly increasing and greater than 1 nm')

    # Build the scene: one custom material, a floor and two area lights.
    world = World()
    Sphere(0.5, world, transform=translate(0, 0.5001, 3.2),
           material=make_material(waves, args.roughness))
    Box(Point3D(-20, -0.1, -20), Point3D(20, 0, 20), world,
        material=Lambert(ConstantSF(0.3)))
    Box(Point3D(-2.5, 3, 1.5), Point3D(-0.5, 3.05, 4.5), world,
        material=UniformSurfaceEmitter(ConstantSF(1), 3))
    Box(Point3D(1.0, 2.5, 2), Point3D(2.2, 2.55, 4.5), world,
        material=UniformSurfaceEmitter(ConstantSF(1), 2))

    # Observe the same scene once per wavelength.
    images = []
    for wavelength in waves:
        pipeline = RadiancePipeline2D(display_progress=False)
        camera = PinholeCamera((args.width, args.height), fov=40, parent=world,
                               transform=translate(0, 2.2, 0) * rotate(0, -27, 0),
                               pipelines=[pipeline])
        camera.render_engine = SerialEngine()
        camera.pixel_samples = args.samples
        camera.spectral_bins = 1
        # Set max first for bands above the observer's default maximum.
        camera.max_wavelength = max(float(wavelength + 1), camera.max_wavelength)
        camera.min_wavelength = float(wavelength - 1)
        camera.max_wavelength = float(wavelength + 1)
        print(f'Rendering {wavelength:g} nm, {args.samples} samples/pixel...', flush=True)
        camera.observe()
        images.append(np.asarray(pipeline.frame.mean)[::-1, ::-1].T.copy())

    # Display and save the renders on a common radiance scale.
    # The 99.5th-percentile white point clips a few brightest pixels.
    vmax = np.percentile(np.concatenate([image.ravel() for image in images]), 99.5)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    for ax, image, wavelength in zip(axes, images, waves):
        im = ax.imshow(image, origin='lower', cmap='gray', vmin=0, vmax=vmax or 1)
        ax.set_title(f'{wavelength:g} nm')
        ax.set_axis_off()
    fig.suptitle(f'One spectral CustomBRDF material; surface roughness {args.roughness:g} µm')
    fig.colorbar(im, ax=axes, label='Band-integrated radiance (W m⁻² sr⁻¹)', fraction=0.025)
    fig.savefig(args.output, dpi=160)
    print(f'Saved {args.output.resolve()}')
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
