from orbitPropagator import (
    cartesianToOrbitalElements,
    orbitalElementsToCartesian,
    twoBodyProp,
)
from orbitPlotter import plotter
import numpy as np
import matplotlib.pyplot as plt

## First we do problem one
G = 6.67 * 10**-11  # N*m^2/kg^2
radius_earth = 6378000  # m
earth_mass = 5.972 * 10**24  # kg

# Orbit A
state_vector_A = [15000000 + radius_earth, 0.3, np.pi / 18, 0, np.pi / 18, 0]

cartesian_state_vector_A = orbitalElementsToCartesian(state_vector_A, earth_mass)

## We want to plot for two orbits
periodA = 2 * (
    2 * np.pi * np.sqrt((15000000 + radius_earth) ** 3 / (G * earth_mass))
)  # seconds
plotter(
    twoBodyProp(cartesian_state_vector_A, oneOrbit=False, timedOrbit=periodA),
    plotEnergy=True,
)
plt.show()

# Orbit B
semilatus_b = -30000000 * (1 - 1.2**2)
true_anomaly_b = np.arccos((semilatus_b - radius_earth) / (radius_earth * 1.2))
print(true_anomaly_b * 180 / np.pi)
state_vector_B = [
    -30000000,
    1.2,
    (80 / 180) * np.pi,
    np.pi,
    np.pi / 18,
    true_anomaly_b,
]
cartesian_state_vector_B = orbitalElementsToCartesian(state_vector_B, earth_mass)

plotter(
    twoBodyProp(cartesian_state_vector_B, oneOrbit=False, timedOrbit=10000),
    plotEnergy=True,
)

plt.show()
