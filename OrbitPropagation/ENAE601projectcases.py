from ENAE601ProjectIntegrator import twoBodyPropRelativistic
from orbitPropagator import cartesianToOrbitalElements, orbitalElementsToCartesian
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sympy import symbols, diff, lambdify
from orbitPropagator import (
    cartesianToOrbitalElements,
    orbitalElementsToCartesian,
)
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker

## Case 1: high mass earth (10x), high eccentricity (0.7) 10000 km height orbit
mu_earth = 3.986 * 10**5
radius_earth = 6371
time_step = 500
semi_major_axis = 10000 + radius_earth  # in km
test_vector = [semi_major_axis, 0.70, np.pi / 2, 0, 0, 0]
cartesian_test_vector = orbitalElementsToCartesian(test_vector, 0, mu=mu_earth)
orbitTest = twoBodyPropRelativistic(
    cartesian_test_vector,
    mu_earth,
    schwarzchild=True,
    desitter=True,
    lensethirring=True,
    oneOrbit=True,
    time_step=time_step,
    export_time=False,
)

# PLot the first and last orbits from orbitTest
first_orbit = orbitTest[0]
last_orbit = orbitTest[-1]
orbitOne = twoBodyPropRelativistic(
    first_orbit,
    mu_earth,
    schwarzchild=False,
    desitter=False,
    lensethirring=False,
    oneOrbit=True,
    time_step=50,
    export_time=False,
)

orbitLast = twoBodyPropRelativistic(
    last_orbit,
    mu_earth,
    schwarzchild=False,
    desitter=False,
    lensethirring=False,
    oneOrbit=True,
    time_step=50,
    export_time=False,
)

# Extract the x, y, and z coordinates for both orbits
x_one = orbitTest[:, 0]
y_one = orbitTest[:, 1]
z_one = orbitTest[:, 2]

x_last = orbitLast[:, 0]
y_last = orbitLast[:, 1]
z_last = orbitLast[:, 2]

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Plot the orbits
ax.plot(x_one, y_one, z_one, label="First Orbit", color="blue", linestyle="--")
# ax.plot(x_last, y_last, z_last, label="Last Orbit", color="red", linestyle="-")

# Labels and title
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
ax.set_title(
    "First and last orbits of 10000km height Orbit about 10x Earth Mass Object Orbiting the Sun Separated by a Month"
)

# Add a legend
ax.legend()

# Show the plot
plt.show()
first_orbitOE = cartesianToOrbitalElements(first_orbit, mu_earth)
last_orbitOE = cartesianToOrbitalElements(last_orbit, mu_earth)

## Set both to periapsis
first_orbitOE[5] = 0
last_orbitOE[5] = 0

## Lets see how far apart they are after thirty days
first_orbitPos = np.array(orbitalElementsToCartesian(first_orbitOE, 0, mu=mu_earth))[:3]
last_orbitPos = np.array(orbitalElementsToCartesian(last_orbitOE, 0, mu=mu_earth))[:3]

distance = np.linalg.norm(last_orbitPos - first_orbitPos)

## Due to relativity after 30 days, periapsis has moved by
print(
    "Due to relativity after 30 days, periapsis has moved by: ", distance * 10**6, "mm"
)
