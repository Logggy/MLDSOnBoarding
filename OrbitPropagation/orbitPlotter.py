from orbitPropagator import twoBodyProp
import matplotlib.pyplot as plt
import numpy as np


def plotter(state_array):
    plt.style.use("dark_background")
    planet_radius = 6.371 * 10**6  # Earth radius meters
    fig = plt.figure(figsize=(10, 10))

    ## Define the axis we will insert our 3d plot of trajectory and planet
    ax = fig.add_subplot(111, projection="3d")

    # Trajectory
    ax.plot(
        state_array[:, 0],
        state_array[:, 1],
        state_array[:, 2],
        "w",
        label="Craft Trajectory",
    )
    ax.plot(
        state_array[0, 0],
        state_array[0, 1],
        state_array[0, 2],
        "wo",
        label="Initial Condition",
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

    ## Now I'll plot the vector of the initial condition
    ax.quiver(
        state_array[0, 0],
        state_array[0, 1],
        state_array[0, 2],
        state_array[0, 3],
        state_array[0, 4],
        state_array[0, 5],
        color="r",
        label="Initial Condition Vector (m / s)",
    )

    ## Set limits + Graph Specs
    graph_limit = np.max(np.abs(state_array[:, :3]))

    ax.set_xlim([-graph_limit, graph_limit])
    ax.set_ylim([-graph_limit, graph_limit])
    ax.set_zlim([-graph_limit, graph_limit])

    ax.set_xlabel(["X (m)"])
    ax.set_ylabel(["Y (m)"])
    ax.set_zlabel(["Z (m)"])

    ax.set_title(["Stable Earth Orbit"])
    plt.legend()
    ## Now that we've got the 3d plot, we can also add the x, y, z, vx, vy, vz 2d plots
    fig2, ax2 = plt.subplots(2, 1, figsize=(10, 10))
    ax_xyz = ax2[0]

    ax_xyz.plot(state_array[:, 0], "w", label="X Position")
    ax_xyz.plot(state_array[:, 1], "r", label="Y Position")
    ax_xyz.plot(state_array[:, 2], "b", label="Z Position")
    ax_xyz.set_xlabel("Time (arbitrary)")
    ax_xyz.set_ylabel("Position (m)")
    ax_xyz.legend()

    ax_vxyz = ax2[1]
    ax_vxyz.plot(state_array[:, 3], "w", label="X Velocity")
    ax_vxyz.plot(state_array[:, 4], "r", label="Y Velocity")
    ax_vxyz.plot(state_array[:, 5], "b", label="Z Velocity")
    ax_vxyz.set_xlabel("Time (arbitrary)")
    ax_vxyz.set_ylabel("Velocity (m / s)")
    ax_vxyz.legend()

    plt.show()


G = 6.67 * 10**-11  # N*m^2/kg^2
m_earth = 5.972 * 10**24  # kg
altitude = 5 * 10**6  # m

plotter(
    twoBodyProp(
        6.371 * 10**6 + altitude,  # radius of the earth plus however many meters
        0,
        0,
        0,
        np.sqrt((G * m_earth) / ((6.371 * 10**6) + altitude)),
        0,
        1000,
    )
)
