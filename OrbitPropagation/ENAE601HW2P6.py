import numpy as np
import matplotlib.pyplot as plt
from orbitPropagator import (
    cartesianToOrbitalElements,
    orbitalElementsToCartesian,
    twoBodyProp,
)
import matplotlib.ticker as mticker

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
    [semimajor_axis_hyper, e_hyper, i, raan, argument_peri, 0],
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
orbitalEEccentric = cartesianToOrbitalElements(
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
print(
    "Orbital elements for ellipse in Keplers orbit Problem: ",
    orbitalEEccentric,
)
print(
    "Cartesian Position vector for ellipse: ",
    rnaughtellipsenew,
    "Cartesian Velocity vector for ellipse: ",
    vnaughtellipsenew,
)


# Repeat for hyperbolic case with tighter convergence criteria
tn, rnaughthypernew, vnaughthypernew = newTimeandR(
    rnaughthyper, vnaughthyper, first_guess_hyper, semimajor_axis_hyper
)
first_guess_hyper = first_guess_hyper + (tof_hyper - tn) / (
    np.linalg.norm(rnaughthyper) / np.sqrt(mu)
)
tries = 1
while (
    np.abs(tn - tof_hyper) > 1e-6
):  # Tighter tolerance for convergence, chat gpt told me to do this and it worked lol
    tn, rnaughthypernew, vnaughthypernew = newTimeandR(
        rnaughthyper, vnaughthyper, first_guess_hyper, semimajor_axis_hyper
    )
    first_guess_hyper = first_guess_hyper + (tof_hyper - tn) / (
        np.linalg.norm(rnaughthypernew) / np.sqrt(mu)
    )
    tries += 1

orbitalEHyper = cartesianToOrbitalElements(
    [
        rnaughthypernew[0] * 10**3,
        rnaughthypernew[1] * 10**3,
        rnaughthypernew[2] * 10**3,
        vnaughthypernew[0] * 10**3,
        vnaughthypernew[1] * 10**3,
        vnaughthypernew[2] * 10**3,
    ],
    mass_earth,
)
print(
    "Orbital elements for hyperbola in Keplers orbit Problem: ",
    orbitalEHyper,
)
print(
    "Cartesian Position vector for hyperbola: ",
    rnaughthypernew,
    "Cartesian Velocity vector for hyperbola: ",
    vnaughthypernew,
)

print(
    "The true anomaly for the elliptic orbit is: ",
    orbitalEEccentric[-1] * 180 / np.pi,
    "degrees",
)
print("The position vector for the elliptic orbit is: ", rnaughtellipsenew, "km")
print("The velocity vector for the elliptic orbit is: ", vnaughtellipsenew, "km/s")
print(
    "The true anomaly for the hyperbolic orbit is: ",
    orbitalEHyper[-1] * 180 / np.pi,
    "degrees",
)
print("The position vector for the hyperbolic orbit is: ", rnaughthypernew, "km")
print("The velocity vector for the hyperbolic orbit is: ", vnaughthypernew, "km/s")

## Now I just need to plot the orbits
ellipse_state_total = twoBodyProp(
    [
        rnaughtellipse[0] * 10**3,
        rnaughtellipse[1] * 10**3,
        rnaughtellipse[2] * 10**3,
        vnaughtellipse[0] * 10**3,
        vnaughtellipse[1] * 10**3,
        vnaughtellipse[2] * 10**3,
    ],
)
planet_mass = (5.972 * 10**24,)
planet_radius = (6.371 * 10**6,)

plt.style.use("dark_background")
fig = plt.figure(figsize=(10, 10))
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))

## Define the axis we will insert our 3d plot of trajectory and planet
ax = fig.add_subplot(111, projection="3d")

# Trajectory
ax.plot(
    ellipse_state_total[:, 0],
    ellipse_state_total[:, 1],
    ellipse_state_total[:, 2],
    "w",
    label="Craft Trajectory",
)
ax.plot(
    rnaughtellipse[0] * 10**3,
    rnaughtellipse[1] * 10**3,
    rnaughtellipse[2] * 10**3,
    "ro",
    label="Initial Condition",
)
ax.plot(
    rnaughtellipsenew[0] * 10**3,
    rnaughtellipsenew[1] * 10**3,
    rnaughtellipsenew[2] * 10**3,
    "bo",
    label="Final Condition",
)

# Host planet
# Pretty easy, using np.mgrid we can make a list of theta and phi values that we can use to convert to cartesian x, y, z values
# keep in mind that the mgrid inputs are styled start:stop:number_of_points where the number_of_points must be a complex value

theta, phi = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
planet_x = planet_radius * np.sin(theta) * np.cos(phi)
planet_y = planet_radius * np.sin(theta) * np.sin(phi)
planet_z = planet_radius * np.cos(theta)

# Plot the planet
ax.plot_surface(planet_x, planet_y, planet_z, cmap="ocean")

## Set limits + Graph Specs
graph_limit = np.max(np.abs(ellipse_state_total[:, :3]))

ax.set_xlim([-graph_limit, graph_limit])
ax.set_ylim([-graph_limit, graph_limit])
ax.set_zlim([-graph_limit, graph_limit])

ax.set_xlabel(["X (m)"])
ax.set_ylabel(["Y (m)"])
ax.set_zlabel(["Z (m)"])

ax.set_title(["Elliptic Orbit With Initial and Final Condition"])
plt.legend()
plt.show()

hyper_state_total = twoBodyProp(
    [
        rnaughthyper[0] * 10**3,
        rnaughthyper[1] * 10**3,
        rnaughthyper[2] * 10**3,
        vnaughthyper[0] * 10**3,
        vnaughthyper[1] * 10**3,
        vnaughthyper[2] * 10**3,
    ],
    oneOrbit=False,
    timedOrbit=1000,
)
planet_mass = (5.972 * 10**24,)
planet_radius = (6.371 * 10**6,)

plt.style.use("dark_background")
fig = plt.figure(figsize=(10, 10))
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))

## Define the axis we will insert our 3d plot of trajectory and planet
ax = fig.add_subplot(111, projection="3d")

# Trajectory
ax.plot(
    hyper_state_total[:, 0],
    hyper_state_total[:, 1],
    hyper_state_total[:, 2],
    "w",
    label="Craft Trajectory",
)
ax.plot(
    rnaughthyper[0] * 10**3,
    rnaughthyper[1] * 10**3,
    rnaughthyper[2] * 10**3,
    "ro",
    label="Initial Condition",
)
ax.plot(
    rnaughthypernew[0] * 10**3,
    rnaughthypernew[1] * 10**3,
    rnaughthypernew[2] * 10**3,
    "bo",
    label="Final Condition",
)

# Host planet
# Pretty easy, using np.mgrid we can make a list of theta and phi values that we can use to convert to cartesian x, y, z values
# keep in mind that the mgrid inputs are styled start:stop:number_of_points where the number_of_points must be a complex value

theta, phi = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
planet_x = planet_radius * np.sin(theta) * np.cos(phi)
planet_y = planet_radius * np.sin(theta) * np.sin(phi)
planet_z = planet_radius * np.cos(theta)

# Plot the planet
ax.plot_surface(planet_x, planet_y, planet_z, cmap="ocean")

## Set limits + Graph Specs
graph_limit = np.max(np.abs(ellipse_state_total[:, :3]))

ax.set_xlim([-graph_limit, graph_limit])
ax.set_ylim([-graph_limit, graph_limit])
ax.set_zlim([-graph_limit, graph_limit])

ax.set_xlabel(["X (m)"])
ax.set_ylabel(["Y (m)"])
ax.set_zlabel(["Z (m)"])

ax.set_title(["Hyperbolic Orbit With Initial and Final Condition"])
plt.legend()
plt.show()
