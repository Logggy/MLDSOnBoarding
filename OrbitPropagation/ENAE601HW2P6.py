import numpy as np
import matplotlib.pyplot as plt
from orbitPropagator import cartesianToOrbitalElements, orbitalElementsToCartesian

# We must solve keplers prediction problem
mass_earth = 5.972 * 10**24  # kg
mu = 3.986 * 10**5
periapsis = 7378
e_ellipse = 0.5
e_hyper = 2
i = 0
raan = 0
argument_peri = 0
true_anomaly_ellipse = 32 * np.pi / 180
true_anomaly_hyper = 0
semimajor_axis_ellipse = periapsis / (1 - e_ellipse)
semimajor_axis_hyper = periapsis / (1 - e_hyper)
period_ellipse = np.pi * 2 * np.sqrt(semimajor_axis_ellipse**3 / mu)
tof_ellipse = period_ellipse * 1 / 3
tof_hyper = 1000
hyper_state_initial = orbitalElementsToCartesian(
    [semimajor_axis_hyper * 10**3, e_hyper, i, raan, argument_peri, 0],
    mass_earth,
    mu=mu,
)
ellipse_state_initial = orbitalElementsToCartesian(
    [
        semimajor_axis_ellipse,
        e_ellipse,
        i,
        raan,
        argument_peri,
        true_anomaly_ellipse,
    ],
    mass_earth,
    mu=mu,
)
first_guess_ellipse = tof_ellipse * np.sqrt(mu) / semimajor_axis_ellipse
first_guess_hyper = np.sqrt(-semimajor_axis_hyper) * np.log(
    (-2 * mu * tof_hyper)
    / (
        semimajor_axis_hyper
        * (
            np.dot(hyper_state_initial[:3], hyper_state_initial[3:])
            + np.sqrt(-mu * semimajor_axis_hyper)
            * (1 - np.linalg.norm(hyper_state_initial[:3]) / semimajor_axis_hyper)
        )
    )
)

## Start with the ellipse
rnaughtellipse = ellipse_state_initial[:3]
vnaughtellipse = ellipse_state_initial[3:]
rnaughthyper = hyper_state_initial[:3]
vnaughthyper = hyper_state_initial[3:]


def csz(chi, semimajor):
    z = chi**2 / semimajor
    if z > 1e-6:
        c = (1 - np.cos(np.sqrt(z))) / z
        s = (np.sqrt(z) - np.sin(np.sqrt(z))) / np.sqrt(z**3)
    elif z < -1e-6:
        c = (1 - np.cosh(np.sqrt(-z))) / z
        s = (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / np.sqrt((-z) ** 3)
    else:
        c, s = 1 / 2, 1 / 6  # When z is close to zero
    return c, s, z


def newTimeandR(rnaught, vnaught, chi, semimajor):
    c, s, z = csz(chi, semimajor)
    # Calculate the time of flight based on the universal anomaly chi
    t = (1 / np.sqrt(mu)) * (
        (chi**3 * s)
        + (np.dot(rnaught, vnaught) * chi**2 * c / np.sqrt(mu))
        + np.linalg.norm(rnaught) * chi * (1 - z * s)
    )
    # Update the new position vector
    f = 1 - (chi**2 * c) / np.linalg.norm(rnaught)
    g = t - (chi**3 * s) / np.sqrt(mu)
    new_r = f * np.array(rnaught) + g * np.array(vnaught)

    f_prime = (
        np.sqrt(mu)
        * chi
        * (z * s - 1)
        * 1
        / (np.linalg.norm(new_r) * np.linalg.norm(rnaught))
    )
    g_prime = 1 - chi**2 * c / np.linalg.norm(new_r)

    # New velocity
    new_v = f_prime * np.array(rnaught) + g_prime * np.array(vnaught)
    return t, new_r, new_v


tn, rnaughtellipsenew, vnaughtellipsenew = newTimeandR(
    rnaughtellipse, vnaughtellipse, first_guess_ellipse, semimajor_axis_ellipse
)
first_guess_ellipse = first_guess_ellipse + (tof_ellipse - tn) / (
    np.linalg.norm(rnaughtellipse) / np.sqrt(mu)
)

tries = 1
while np.abs(tn - tof_ellipse) > 1:
    tn, rnaughtellipsenew, vnaughtellipsenew = newTimeandR(
        rnaughtellipse, vnaughtellipse, first_guess_ellipse, semimajor_axis_ellipse
    )
    first_guess_ellipse = first_guess_ellipse + (tof_ellipse - tn) / (
        np.linalg.norm(rnaughtellipsenew) / np.sqrt(mu)
    )

    tries += 1

print(
    cartesianToOrbitalElements(
        [
            rnaughtellipsenew[0] * 10**3,
            rnaughtellipsenew[1] * 10**3,
            rnaughtellipsenew[2] * 10**3,
            vnaughtellipsenew[0] * 10**3,
            vnaughtellipsenew[1] * 10**3,
            vnaughtellipsenew[2] * 10**3,
        ],
        mass_earth,
    )
)
print(rnaughtellipsenew, vnaughtellipsenew)
