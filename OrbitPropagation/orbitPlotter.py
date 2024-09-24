from orbitPropagator import (
    twoBodyProp,
    cartesianToOrbitalElements,
    orbitalElementsToCartesian,
)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def plotter(
    state_array,
    plotEnergy=False,
    planet_mass=5.972 * 10**24,
    planet_radius=6.371 * 10**6,
    time_step=10,
):
    radius_earth = 6378000  # m
    G = 6.67 * 10**-11  # N*m^2/kg^2

    # save this to compute accelerations
    def twoBodyDifEq(state_vector, planet_mass):
        r_norm = np.linalg.norm(state_vector[:3])
        ax, ay, az = -(G * planet_mass * state_vector[:3]) / (r_norm**3)

        return [state_vector[3], state_vector[4], state_vector[5], ax, ay, az]

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(10, 10))
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))

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

    # Periapsis is the same location as true anomaly = 0

    orbital_state = cartesianToOrbitalElements(state_array[0], planet_mass)

    orbital_state[5] = 0

    periapsis_cartesian = orbitalElementsToCartesian(orbital_state, planet_mass)

    ax.plot(
        periapsis_cartesian[0],
        periapsis_cartesian[1],
        periapsis_cartesian[2],
        "ro",
        label="Periapsis",
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
    time_array = time_step * np.arange(len(state_array))
    ## Now that we've got the 3d plot, we can also add the x, y, z, vx, vy, vz 2d plots
    if plotEnergy:
        fig2, ax2 = plt.subplots(3, 1, figsize=(10, 10))
        ax_xyz = ax2[0]
        r_mag = np.zeros(len(state_array))
        v_mag = np.zeros(len(state_array))
        a_mag = np.zeros(len(state_array))

        for i in range(len(r_mag)):
            r_mag[i] = np.sqrt(
                state_array[i, 0] ** 2 + state_array[i, 1] ** 2 + state_array[i, 2] ** 2
            )
            v_mag[i] = np.sqrt(
                state_array[i, 3] ** 2 + state_array[i, 4] ** 2 + state_array[i, 5] ** 2
            )
            acceleration_array = twoBodyDifEq(state_array[i], planet_mass)[3:]
            a_mag[i] = np.sqrt(
                acceleration_array[0] ** 2
                + acceleration_array[1] ** 2
                + acceleration_array[2] ** 2
            )
        # find the index of r_mag_min
        Orbit_A_Period = (
            2 * np.pi * np.sqrt((15000000 + radius_earth) ** 3 / (G * planet_mass))
        )  # seconds
        r_min_index = np.argmin(r_mag)
        ax_xyz.set_title("Plots of Position, Velocity, and Acceleration Magnitudes")
        ax_xyz.plot(time_array, r_mag, "w", label="Position Mag")
        ax_xyz.plot(
            [
                time_array[r_min_index],
                time_array[r_min_index] + Orbit_A_Period,
                time_array[r_min_index] + 2 * Orbit_A_Period,
            ],
            [r_mag[r_min_index], r_mag[r_min_index], r_mag[r_min_index]],
            "ro",
            label="Periapsis",
        )
        ax_xyz.set_xlabel("Time (s)")
        ax_xyz.set_ylabel("Position Magnitude (m)")
        ax_xyz.legend()

        ax_vxyz = ax2[1]
        ax_vxyz.plot(time_array, v_mag, "w", label="Velocity Mag")
        ax_vxyz.set_xlabel("Time (s)")
        ax_vxyz.set_ylabel("Velocity Magnitude (m / s)")
        ax_vxyz.legend()

        ax_axyz = ax2[2]
        ax_axyz.plot(time_array, a_mag, "w", label="Acceleration Mag")
        ax_axyz.set_xlabel("Time (s)")
        ax_axyz.set_ylabel("Acceleration Magnitude (m / s^2)")
        ax_axyz.legend()
    else:
        fig2, ax2 = plt.subplots(2, 1, figsize=(10, 10))
        ax_xyz = ax2[0]

        ax_xyz.plot(time_array, state_array[:, 0], "w", label="X Position")
        ax_xyz.plot(time_array, state_array[:, 1], "r", label="Y Position")
        ax_xyz.plot(time_array, state_array[:, 2], "b", label="Z Position")
        ax_xyz.set_xlabel("Time (s)")
        ax_xyz.set_ylabel("Position (m)")
        ax_xyz.legend()

        ax_vxyz = ax2[1]
        ax_vxyz.plot(time_array, state_array[:, 3], "w", label="X Velocity")
        ax_vxyz.plot(time_array, state_array[:, 4], "r", label="Y Velocity")
        ax_vxyz.plot(time_array, state_array[:, 5], "b", label="Z Velocity")
        ax_vxyz.set_xlabel("Time (s)")
        ax_vxyz.set_ylabel("Velocity (m / s)")
        ax_vxyz.legend()

    ## Here I also need to add the option for plotting energy - which will also plot the r, v, h and osculating orbital elements
    if plotEnergy:
        fig3, ax3 = plt.subplots(1, 1, figsize=(10, 10))
        ax_energy = ax3
        ax_energy.yaxis.set_major_formatter(formatter)
        initial_kinetic = (
            state_array[0, 3] ** 2 + state_array[0, 4] ** 2 + state_array[0, 5] ** 2
        ) * 0.5
        initial_potential = -(planet_mass * G) / np.sqrt(
            state_array[0, 0] ** 2 + state_array[0, 1] ** 2 + state_array[0, 2] ** 2
        )
        initial_energy = initial_kinetic + initial_potential
        energy = np.zeros(len(state_array))
        for i in range(len(energy)):
            ## Energy is Kinetic plus Potential
            kinetic = (
                state_array[i, 3] ** 2 + state_array[i, 4] ** 2 + state_array[i, 5] ** 2
            ) * 0.5
            potential = -(planet_mass * G) / np.sqrt(
                state_array[i, 0] ** 2 + state_array[i, 1] ** 2 + state_array[i, 2] ** 2
            )
            energy[i] = (kinetic + potential) - initial_energy
        ax_energy.set_title("Time Varying Deviation in Orbital Energy")
        ax_energy.plot(time_array, energy, "w", label="Total E")
        ax_energy.set_xlabel("Time (s)")
        ax_energy.set_ylabel("Energy - Initial Energy (J / kg)")
        ax_energy.legend()
        fig4, ax4 = plt.subplots(3, 1, figsize=(10, 10))
        angular_momentum = np.zeros((len(state_array), 3))
        for i in range(len(angular_momentum)):
            h = np.cross(state_array[i, :3], state_array[i, 3:])
            angular_momentum[i, 0] = h[0]
            angular_momentum[i, 1] = h[1]
            angular_momentum[i, 2] = h[2]

        ax_momentumi = ax4[0]
        ax_momentumi.yaxis.set_major_formatter(formatter)
        ax_momentumi.set_title("Angular momentum in three dimensions")
        ax_momentumi.plot(
            time_array, angular_momentum[:, 0], "w", label="Angular Momentum i"
        )
        ax_momentumi.set_xlabel("Time (s)")
        ax_momentumi.set_ylabel("Specific Angular Momentum (m^2 / s)")
        ax_momentumi.legend()

        ax_momentumj = ax4[1]
        ax_momentumj.yaxis.set_major_formatter(formatter)
        ax_momentumj.plot(
            time_array, angular_momentum[:, 1], "w", label="Angular Momentum j"
        )
        ax_momentumj.set_xlabel("Time (s)")
        ax_momentumj.set_ylabel("Specific Angular Momentum (m^2 / s)")
        ax_momentumj.legend()

        ax_momentumk = ax4[2]
        ax_momentumk.yaxis.set_major_formatter(formatter)
        ax_momentumk.plot(
            time_array, angular_momentum[:, 2], "w", label="Angular Momentum k"
        )
        ax_momentumk.set_xlabel("Time (s)")
        ax_momentumk.set_ylabel("Specific Angular Momentum (m^2 / s)")
        ax_momentumk.legend()

        fig5, ax5 = plt.subplots(6, 1, figsize=(30, 30))
        ## Now the osculating orbital elements
        a = np.zeros(len(state_array))
        e = np.zeros(len(state_array))
        inc = np.zeros(len(state_array))
        Omega = np.zeros(len(state_array))
        omega = np.zeros(len(state_array))
        nu = np.zeros(len(state_array))
        for i in range(len(state_array)):
            orbital_state_osculating = cartesianToOrbitalElements(
                state_array[i], planet_mass
            )
            a[i] = orbital_state_osculating[0]
            e[i] = orbital_state_osculating[1]
            inc[i] = orbital_state_osculating[2]
            Omega[i] = orbital_state_osculating[3]
            omega[i] = orbital_state_osculating[4]
            nu[i] = orbital_state_osculating[5]
        ax5[0].yaxis.set_major_formatter(formatter)
        ax5[1].yaxis.set_major_formatter(formatter)
        ax5[2].yaxis.set_major_formatter(formatter)
        ax5[3].yaxis.set_major_formatter(formatter)
        ax5[4].yaxis.set_major_formatter(formatter)
        ax5[5].yaxis.set_major_formatter(formatter)
        ax_a = ax5[0]
        ax_a.set_title("Osculating Orbital Elements")
        ax_a.plot(time_array, a, "w", label="semimajor axis")
        ax_a.set_xlabel("Time (s)")
        ax_a.set_ylabel("Semi-major Axis (m)")
        ax_a.legend()

        ax_e = ax5[1]
        ax_e.plot(time_array, e, "w", label="eccentricity")
        ax_e.set_xlabel("Time (s)")
        ax_e.set_ylabel("Eccentricity")
        ax_e.legend()

        ax_inc = ax5[2]
        ax_inc.plot(time_array, inc, "w", label="inclination")
        ax_inc.set_xlabel("Time (s)")
        ax_inc.set_ylabel("Inclination (rad)")
        ax_inc.legend()

        ax_Omega = ax5[3]
        ax_Omega.plot(time_array, Omega, "w", label="Omega")
        ax_Omega.set_xlabel("Time (s)")
        ax_Omega.set_ylabel("RAAN (rad)")
        ax_Omega.legend()

        ax_omega = ax5[4]
        ax_omega.plot(time_array, omega, "w", label="arg_peri")
        ax_omega.set_xlabel("Time (s)")
        ax_omega.set_ylabel("Argument of Periapsis (rad)")
        ax_omega.legend()

        ax_nu = ax5[5]
        ax_nu.plot(time_array, nu, "w", label="anomaly")
        ax_nu.set_xlabel("Time (s)")
        ax_nu.set_ylabel("true Anomaly (rad)")
        ax_nu.legend()


# G = 6.67 * 10**-11  # N*m^2/kg^2
# m_earth = 5.972 * 10**24  # kg
# altitude = 5 * 10**6  # m

# plotter(
#     twoBodyProp(
#         6.371 * 10**6 + altitude,  # radius of the earth plus however many meters
#         0,
#         0,
#         0,
#         np.sqrt((G * m_earth) / ((6.371 * 10**6) + altitude)),
#         0,
#         1000,
#     )
# )

# plt.show()
