import math
import numpy as np
import matplotlib.pyplot as plt


def total_internal_reflection_angle(n1, n2):
    if n1 > n2:
        n1, n2 = n2, n1
    return transmission_angle(n1, n2, 0.5 * math.pi)


def transmission_angle(n1, n2, angle):
    return math.asin(n1 * math.sin(angle) / n2)


def brewster_angle(n1, n2):
    if n1 > n2:
        n1, n2 = n2, n1
    return math.atan(n2 / n1)


def fresnel_rp(n1, n2, angle):
    """
    Fresnel reflection coefficient for parallel E-field.
    """

    if n1 > n2 and angle > total_internal_reflection_angle(n1, n2):
        return 1.0

    ci = math.cos(angle)
    ct = math.cos(transmission_angle(n1, n2, angle))
    return (n1*ct - n2*ci) / (n2*ci + n1*ct)


def fresnel_rs(n1, n2, angle):
    """
    Fresnel reflection coefficient for perpendicular E-field.
    """

    if n1 > n2 and angle > total_internal_reflection_angle(n1, n2):
        return 1.0

    ci = math.cos(angle)
    ct = math.cos(transmission_angle(n1, n2, angle))
    return (n1*ci - n2*ct) / (n1*ci + n2*ct)


def fresnel_tp(n1, n2, angle):
    """
    Fresnel transmission coefficient for parallel E-field.
    """

    if n1 > n2 and angle > total_internal_reflection_angle(n1, n2):
        return 0.0

    t = transmission_angle(n1, n2, angle)
    ci = math.cos(angle)
    ct = math.cos(t)
    return 2*n1*ci / (n2*ci + n1*ct)


def fresnel_ts(n1, n2, angle):
    """
    Fresnel transmission coefficient for perpendicular E-field.
    """

    if n1 > n2 and angle > total_internal_reflection_angle(n1, n2):
        return 0.0

    t = transmission_angle(n1, n2, angle)
    ci = math.cos(angle)
    ct = math.cos(t)
    return 2*n1*ci / (n1*ci + n2*ct)


# optimised coefficient calculations
def fresnel_non_tir(ni, nt, angle):

    if ni > nt and angle > total_internal_reflection_angle(ni, nt):
        return 1.0, 1.0, 0.0, 0.0, 0.0

    ci = math.cos(angle)
    ct = math.cos(transmission_angle(ni, nt, angle))

    # common coefficients
    k0 = ni * ct
    k1 = nt * ci
    k2 = ni * ci
    k3 = nt * ct

    a = 1.0 / (k1 + k0)
    b = 1.0 / (k2 + k3)

    # reflection coefficients
    rp = a * (k0 - k1)
    rs = b * (k2 - k3)

    # transmission coefficients
    tp = 2 * a * k2
    ts = 2 * b * k2

    # projected area for transmitted beam
    ta = k3 / k2

    return rp, rs, tp, ts, ta


# # equation according to Stellar Polarimetry
# def tir_phase_p(n1, n2, angle):
#
#     if n1 < n2 or angle < total_internal_reflection_angle(n1, n2):
#         return 0.0
#
#     ci = math.cos(angle)
#     gamma = n1 / n2
#     return 2 * math.atan2(gamma * math.sqrt(gamma**2*(1 - ci**2) - 1), ci) - math.pi
#
#
# # equation according to Stellar Polarimetry
# def tir_phase_s(n1, n2, angle):
#
#     if n1 < n2 or angle < total_internal_reflection_angle(n1, n2):
#         return 0.0
#
#     ci = math.cos(angle)
#     gamma = n1 / n2
#     return 2 * math.atan2(math.sqrt(gamma**2*(1 - ci**2) - 1), gamma * ci)
#
#
# # equation according to Stellar Polarimetry
# def fresnel_tir_phase(ni, nt, angle):
#
#     if ni < nt or angle < total_internal_reflection_angle(ni, nt):
#         return 0.0
#
#     ci = math.cos(angle)
#     gamma = ni / nt
#     k = gamma*gamma*(1 - ci*ci) - 1
#     return math.pi + 2 * math.atan2(ci*(1 - gamma*gamma)*math.sqrt(k), gamma*(ci*ci + k))


# equation according to Polarized Light by Goldstein
def tir_phase_p(n1, n2, angle):

    if n1 < n2 or angle < total_internal_reflection_angle(n1, n2):
        return 0.0

    ci = math.cos(angle)
    gamma = n1 / n2
    return 2 * math.atan2(gamma * math.sqrt(gamma**2*(1 - ci**2) - 1), ci)


# equation according to Polarized Light by Goldstein
def tir_phase_s(n1, n2, angle):

    if n1 < n2 or angle < total_internal_reflection_angle(n1, n2):
        return 0.0

    ci = math.cos(angle)
    gamma = n1 / n2
    return 2 * math.atan2(math.sqrt(gamma**2*(1 - ci**2) - 1), gamma * ci)


# equation according to Polarized Light by Goldstein
def fresnel_tir_phase(n1, n2, angle):

    if n1 < n2 or angle < total_internal_reflection_angle(n1, n2):
        return 0.0

    ci = math.cos(angle)
    gamma = n1 / n2
    return -2 * math.atan2(ci*math.sqrt(gamma**2*(1 - ci**2) - 1), gamma*(1-ci**2))




def plot_fresnel(n1, n2):

    n = 1000
    a = np.linspace(0, 90, n)
    rp = np.zeros(n)
    rs = np.zeros(n)
    tp = np.zeros(n)
    ts = np.zeros(n)
    Rp = np.zeros(n)
    Rs = np.zeros(n)
    Tp = np.zeros(n)
    Ts = np.zeros(n)
    pp = np.zeros(n)
    ps = np.zeros(n)
    P = np.zeros(n)
    for i in range(n):

        incident = math.radians(a[i])
        rp[i], rs[i], tp[i], ts[i], ta = fresnel_non_tir(n1, n2, incident)
        Rp[i] = rp[i]**2            # beam area for reflected and incident identical, i.e. area normalisation = 1.0
        Rs[i] = rs[i]**2            # beam area for reflected and incident identical, i.e. area normalisation = 1.0
        Tp[i] = (tp[i]**2) * ta     # normalise with projected beam area to give flux
        Ts[i] = (ts[i]**2) * ta     # normalise with projected beam area to give flux
        pp[i] = math.degrees(tir_phase_p(n1, n2, incident))
        ps[i] = math.degrees(tir_phase_s(n1, n2, incident))
        P[i] = math.degrees(fresnel_tir_phase(n1, n2, incident))

    plt.figure()
    plt.title(f'Coefficients n1={n1}, n2={n2}')
    plt.grid()
    plt.plot(a, rp)
    plt.plot(a, rs)
    plt.plot(a, tp)
    plt.plot(a, ts)
    plt.legend(['rp', 'rs', 'tp', 'ts'])

    plt.figure()
    plt.title(f'Reflection/transmission n1={n1}, n2={n2}')
    plt.grid()
    plt.plot(a, Rp)
    plt.plot(a, Rs)
    plt.plot(a, 0.5*(Rp + Rs))
    plt.plot(a, Tp)
    plt.plot(a, Ts)
    plt.plot(a, 0.5*(Tp + Ts))
    plt.plot(a, 0.5*(Rp + Rs) + 0.5*(Tp + Ts))
    plt.legend(['Rp', 'Rs', 'R', 'Tp', 'Ts', 'T', 'TOTAL'])

    plt.figure()
    plt.title(f'TIR Reflection Phase n1={n1}, n2={n2}')
    plt.grid()
    plt.plot(a, pp)
    plt.plot(a, ps)
    plt.plot(a, ps - pp)
    plt.plot(a, P)
    plt.legend(['pp', 'ps', 'ps - pp', 'P'])

    plt.figure()
    plt.title(f'TIR Mueller Coeffs n1={n1}, n2={n2}')
    plt.grid()
    plt.plot(a, np.cos(np.radians(P)))
    plt.plot(a, np.sin(np.radians(P)))
    plt.legend(['cos', 'sin'])


from raysect.core import Vector3D

i = Vector3D(-0.1, -1, 0).normalise()
n = Vector3D(0, 1, 0).normalise()

g = 1.5 / 1.0
ci = abs(n.dot(i))
ct = math.sqrt(1 - (g * g) * (1 - ci * ci))
print(f'ci={math.degrees(math.acos(ci))}, ct={math.degrees(math.acos(ct))}')

temp = g * ci - ct
t = Vector3D(
    g * i.x + temp * n.x,
    g * i.y + temp * n.y,
    g * i.z + temp * n.z
)

print(f'n={n}, i={i}, t={t}')
print(f'cos(t.n)={math.degrees(math.acos(t.dot(-n)))}')

# plot_fresnel(1.0, 1.5)
# plot_fresnel(1.5, 1.0)
# plot_fresnel(1.5151, 1.0)
# plot_fresnel(4, 1)
# plot_fresnel(3, 2)
# plt.show()
