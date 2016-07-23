from matplotlib.pylab import *
from raysect.core import Vector3D
from random import random


def d_vector(s_half, roughness):

    # ggx distribution
    r2 = roughness * roughness
    h2 = s_half.z * s_half.z
    k = h2 * (r2 - 1) + 1
    return r2 / (pi * k * k)


def d_angle(theta, roughness):
    r2 = roughness * roughness
    return r2 / (pi * cos(theta)**4 * (r2 + tan(theta)**2)**2)


def d_sample(roughness):

    e1 = random()
    e2 = random()

    theta = arctan(roughness * sqrt(e1) / sqrt(1 - e1))
    phi = 2*pi*e2

    z = cos(theta)
    x = cos(phi) * sin(theta)
    y = sin(phi) * sin(theta)

    return Vector3D(x, y, z)


def outgoing_sample(incoming, facet_normal):
    return 2 * incoming.dot(facet_normal) * facet_normal - incoming


def pdf(incoming, outgoing, roughness):

    h = ((incoming + outgoing) * 0.5).normalise()
    return 0.25 * d_vector(h, roughness) / abs(outgoing.dot(h)) * abs(h.z)

# a = linspace(-pi/2, pi/2, 200)
#
# for r in linspace(0.1, 1, 25):
#
#     # figure(1)
#     # plot(a / pi * 180, d_angle(a, r))
#     #
#     # figure(2)
#     # plot(a / pi * 180, d_angle(a, r) * cos(a))
#
#     incoming = 30
#     h = (a + (incoming / 180 * pi)) / 2
#
#     figure(3)
#     plot(a / pi * 180, d_angle(h, r) * cos(h) / (4 * cos(a - h)))
#
#     figure(4)
#     w = pow(cos(incoming / 180 * pi), 1/3)
#     r_mod = r * w + (1 - w)
#     plot(a / pi * 180, d_angle(h, r_mod) * cos(h) / (4 * cos(a - h)))


# GGX distribution plot
incoming = Vector3D(1,0,1).normalise()
rmin = 0.01
rmax = 1.0
n = 9
s = ceil(sqrt(n))
for f, r in enumerate(linspace(rmin, rmax, n)):

    r **= 2
    print("generating distribution, roughness={}".format(r))
    normal = []
    out = []
    for i in range(25000):

        facet_normal = d_sample(r)
        normal.append((facet_normal.x, facet_normal.y, facet_normal.z, d_vector(facet_normal, r)))

        outgoing = outgoing_sample(incoming, facet_normal)
        out.append((outgoing.x, outgoing.y, outgoing.z, pdf(incoming, outgoing, r)))

    normal = array(normal)
    out = array(out)
    mask = out[:, 1] < 5
    # mask = (out[:, 1] > -0.1) & (out[:, 1] < 0.1)

    # figure(100)
    # subplot(s, s, f+1)
    # plot(normal[:, 0], normal[:, 2], '.')
    # plot(out[:, 0], out[:, 2], 'x')
    # title(r)
    # xlim(-1, 1)
    # ylim(-1, 1)

    # figure(201)
    # subplot(s, s, f+1)
    # scatter(normal[:, 0], normal[:, 2], c=normal[:,3])
    # title("normal " + str(r))
    # xlim(-1, 1)
    # ylim(-1, 1)
    #
    # figure(202)
    # subplot(s, s, f+1)
    # scatter(out[:, 0], out[:, 2], c=out[:,3])
    # title("out " + str(r))
    # xlim(-1, 1)
    # ylim(-1, 1)

    figure(102)
    subplot(s, s, f+1)
    hist2d(out[mask,0], out[mask,2], bins=50, range=[[-1, 1], [-1, 1]])
    title(r)
    xlim(-1, 1)
    ylim(-1, 1)

    figure(103)
    subplot(s, s, f + 1)
    scatter(out[mask,0], out[mask,2], c=out[mask,3], edgecolors='none', s=out[mask,3] / out[mask,3].max() * 20)
    title(r)
    xlim(-1, 1)
    ylim(-1, 1)
    #
    # figure(104)
    # mask = normal[:, 2] > 0
    # subplot(s, s, f + 1)
    # scatter(normal[mask,0], normal[mask,1], c=normal[mask,3])
    # title(r)
    # xlim(-1, 1)
    # ylim(-1, 1)

show()




