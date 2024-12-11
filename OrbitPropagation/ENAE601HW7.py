import numpy as np
import matplotlib.pyplot as plt
from orbitPropagator import (
    twoBodyProp,
    cartesianToOrbitalElements,
    orbitalElementsToCartesian,
)
from orbitPlotter import plotter


## Problem 1
## We'll start by getting the two body solutions

radius_earth = 6371  ## km
mu_earth = 3.986 * 10**5
atarget = (radius_earth + 1010) / (1 - 0)
ainter = (radius_earth + 1000) / (1 - 0.15)
orbit1OE = [
    atarget,
    0,
    40 * np.pi / 180,
    25 * np.pi / 180,
    15 * np.pi / 180,
    0,
]
orbit2OE = [
    ainter,
    0.15,
    40 * np.pi / 180,
    25 * np.pi / 180,
    15 * np.pi / 180,
    0,
]

orbit1Ct = np.array(orbitalElementsToCartesian(orbit1OE, 0, mu=mu_earth))
orbit2Ct = np.array(orbitalElementsToCartesian(orbit2OE, 0, mu=mu_earth))

orbit1 = twoBodyProp(orbit1Ct, mu_earth)
orbit2 = twoBodyProp(orbit2Ct, mu_earth)

orbit1r = np.array(orbit1)[:, :3]
orbit2r = np.array(orbit2)[:, :3]

## Plot these two orbits to make sure everything is fine
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

# Plot the Earth too
theta, phi = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
planet_x = radius_earth * np.sin(theta) * np.cos(phi)
planet_y = radius_earth * np.sin(theta) * np.sin(phi)
planet_z = radius_earth * np.cos(theta)

# Plot the planet
ax.plot_surface(planet_x, planet_y, planet_z, cmap="ocean")

ax.plot(
    orbit1r[:, 0],
    orbit1r[:, 1],
    orbit1r[:, 2],
    label="Target",
    color="green",
)
ax.plot(
    orbit2r[:, 0],
    orbit2r[:, 1],
    orbit2r[:, 2],
    label="Interceptor",
    color="red",
)

ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
ax.set_title("3D Orbit Plot")
ax.legend()

ax.set_box_aspect([1, 1, 1])
plt.show()


## Now I need to make an Cartesian - RSW converter
def carttoRSW(cartesian_state_vector):
    r = cartesian_state_vector[:3]
    v = cartesian_state_vector[3:]

    # first we make the unit vectors in the RSW direction
    R_unit = r / np.linalg.norm(r)
    S_unit = v / np.linalg.norm(v)
    # W is in the direction of angular momentum
    W_unit = np.cross(r, v) / np.linalg.norm(np.cross(r, v))

    rotation_matrix = np.vstack([R_unit, S_unit, W_unit])

    # now just rotate the position and velocity
    RSW_r = np.dot(rotation_matrix, r)
    RSW_v = np.dot(rotation_matrix, v)

    return RSW_r, RSW_v


## Ok now that we have that we'll want to get the relative positions using integration and CW equations
half_period = np.pi * np.sqrt(orbit1OE[0] ** 3 / mu_earth)
time_step = 10
orbit1half = twoBodyProp(
    orbit1Ct, mu_earth, oneOrbit=False, timedOrbit=half_period, time_step=time_step
)
orbit2half = twoBodyProp(
    orbit2Ct, mu_earth, oneOrbit=False, timedOrbit=half_period, time_step=time_step
)
steps = 1000
half_period_time = np.linspace(0, half_period, steps)

R = np.zeros(len(half_period_time))
S = np.zeros(len(half_period_time))
W = np.zeros(len(half_period_time))

r_vec, v_vec = carttoRSW(orbit2Ct - orbit1Ct)


x0, y0, z0 = r_vec
x00, y00, z00 = v_vec
for i in range(steps):
    omega_target = np.sqrt(mu_earth / orbit1OE[0] ** 3)
    R[i] = (
        (4 * x0)
        + (2 * y00 / omega_target)
        + ((x00 / omega_target) * np.sin(omega_target * half_period_time[i]))
        - (
            ((2 * y00 / omega_target) + (3 * x0))
            * np.cos(omega_target * half_period_time[i])
        )
    )
    S[i] = (
        ((2 * x00 / omega_target) * np.cos(omega_target * half_period_time[i]))
        + (
            ((4 * y00 / omega_target) + (6 * x0))
            * np.sin(omega_target * half_period_time[i])
        )
        + (((-6 * omega_target * x0) - (3 * y00)) * half_period_time[i])
        + y0
        - (2 * x00 / omega_target)
    )
    W[i] = (z0 * np.cos(omega_target * half_period_time[i])) + (
        (z00 / omega_target) * np.sin(omega_target * half_period_time[i])
    )


## Now I should check if this is correct
# plt.plot(half_period_time, R, label="R")
# plt.plot(half_period_time, S, label="S")
# plt.plot(half_period_time, W, label="W")
# plt.show()
## Seems alright I guess
print(R[0], S[0], W[0])
true_diff = np.zeros((len(orbit1half), 3))
time_prop = np.linspace(0, half_period, len(orbit1half))

for i in range(len(orbit1half)):
    RSW_prop, RSW_propv = carttoRSW(np.array(orbit2half[i]) - np.array(orbit1half[i]))
    true_diff[i] = RSW_prop
print(np.array(orbit2half[-1]) - np.array(orbit1half[-1]))
## After getting the true differences I also need to point them in the right directions
print(true_diff[0])
fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

# Plot R
axs[0].plot(half_period_time, R, label="CW R")
axs[0].plot(time_prop, [row[0] for row in true_diff], label="Propagated R")
axs[0].set_ylabel("Distance (km)")
axs[0].set_title("RSW Values from CW equations vs True Propagation")
axs[0].legend()

# Plot S
axs[1].plot(half_period_time, S, label="CW S")
axs[1].plot(time_prop, [row[1] for row in true_diff], label="Propagated S")
axs[1].set_ylabel("Distance (km)")
axs[1].set_title("RSW Values from CW equations vs True Propagation")
axs[1].legend()

# Plot W
axs[2].plot(half_period_time, W, label="CW W")
axs[2].plot(time_prop, [row[2] for row in true_diff], label="Propagated W")
axs[2].set_ylabel("Distance (km)")
axs[2].set_xlabel("Time (s)")
axs[2].set_title("RSW Values from CW equations vs True Propagation")
axs[2].legend()

plt.show()


## now see when CW exceeds 1% target position
## I would say that this means the magnitude of the difference should be 1% of the magnitude of the target spacecrafts
## distance from earth

percenttime = 0
for i in range(len(half_period_time)):
    if np.linalg.norm([R[i], S[i], W[i]]) > np.linalg.norm(orbit1half[i, :3]) * 0.01:
        percenttime = half_period_time[i]
        break

## Our value is: 140s
print(percenttime)

## Confirmed that my stuff is looking good!
## We know that our CW and propagated equations are looking good because W is basically 0 as it should be, since they are inclined the same,
## we know that there should be no difference between the two in the direction of orbital angular momentum.

## Question 2
final_time = 100
omega_target_final = np.sqrt(mu_earth / orbit1OE[0] ** 3) * final_time
omega_target = np.sqrt(mu_earth / orbit1OE[0] ** 3)
matrixRRtf = np.matrix(
    [
        [4 - 3 * np.cos(omega_target_final), 0, 0],
        [6 * (np.sin(omega_target_final) - omega_target_final), 1, 0],
        [0, 0, np.cos(omega_target_final)],
    ]
)

matrixRVtf = np.matrix(
    [
        [
            np.sin(omega_target_final) / omega_target,
            (2 / omega_target) * (1 - np.cos(omega_target_final)),
            0,
        ],
        [
            (-2 / omega_target) * (1 - np.cos(omega_target_final)),
            (4 * np.sin(omega_target_final) - 3 * omega_target_final) / omega_target,
            0,
        ],
        [0, 0, np.sin(omega_target_final) / omega_target],
    ]
)

matrixVRtf = np.matrix(
    [
        [3 * omega_target * np.sin(omega_target_final), 0, 0],
        [6 * (omega_target * np.cos(omega_target_final) - omega_target), 0, 0],
        [0, 0, -omega_target * np.sin(omega_target_final)],
    ]
)

matrixVVtf = np.matrix(
    [
        [np.cos(omega_target_final), 2 * np.sin(omega_target_final), 0],
        [-2 * np.sin(omega_target_final), 4 * np.cos(omega_target_final) - 3, 0],
        [0, 0, np.cos(omega_target_final)],
    ]
)

## R_0 is defined above as r_vec

v_t0 = -np.linalg.inv(matrixRVtf) @ matrixRRtf @ r_vec
v_tf = (matrixVRtf - matrixVVtf @ np.linalg.inv(matrixRVtf) @ matrixRRtf) @ r_vec

print("The relative vector (RSW) for the first maneuver is: ", v_t0 - v_vec, " km/s")
print("The relative vector (RSW) for the second maneuver is: ", -v_tf, " km/s")
