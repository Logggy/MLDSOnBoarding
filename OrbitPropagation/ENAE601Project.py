import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from orbitPropagator import (
    cartesianToOrbitalElements,
    orbitalElementsToCartesian,
    twoBodyProp,
)
from matplotlib.colors import Normalize

## First I just need to establish a propagator that includes gravitational precession


def twoBodyPropRelativistic(
    cartesian_state_vector,
    mu,
    schwarzchild=False,
    lensethirring=False,
    desitter=False,
    time_step=10,
    export_time=True,
    oneOrbit=False,
    timedOrbit=10,
):
    ## Establish State
    x = cartesian_state_vector[0]
    y = cartesian_state_vector[1]
    z = cartesian_state_vector[2]
    vx = cartesian_state_vector[3]
    vy = cartesian_state_vector[4]
    vz = cartesian_state_vector[5]
    ## Contemplate adding an argument for example initial conditions!
    ## Lets establish some constants
    initial_state_vector = [x, y, z, vx, vy, vz]
    ## Now we propagate
    ## In this solution we know that F = GMm/r^2
    ## Essentially we already know the solutions to Newtons Equations namely - a = -GMr / mag_r^3, where a and r are vectors
    ## So now it's easy - we have the solutions of the original function (x, y, z, vx, vy, vz)
    ## AND their derivatives (vx, vy, vz, ax, ay, az), all we need to do is integrate to solve the ODE
    ## We are applying an acceleration in the direction of the velocity vector, so we can just add this in within the difeq itself
    a_s_array = []
    a_d_array = []
    a_lt_array = []

    a_s_magarray = []
    a_d_magarray = []
    a_lt_magarray = []

    def twoBodyDifEq(t, state_vector, mu):
        r_vector = state_vector[:3]
        v_vector = state_vector[3:]
        r_norm = np.linalg.norm(state_vector[:3])
        ax, ay, az = -(mu * state_vector[:3]) / (r_norm**3)
        v_norm = np.linalg.norm(state_vector[3:])
        ## Fortunately the magnitudes of these three relativistic accelerations are well described in
        ## Sosnica et al 2021, thank you!
        ## We only need to add a few more terms here: mu_sun, R_s and Rdot_s
        ## R_s = position of the earth with respect to the sun
        ## Rdot_s = velocity of the earth relative to the sun
        ## To do this I will manually evaluate the change in earths true anomaly assuming a circular orbit
        ## Earths true anomaly starts at zero and we know we change 2pi radians per 365 days
        ## Assuming earth starts at some arbitrary true anomaly zero:
        true_anomaly_earth_rad = (
            2 * np.pi / (3.154 * 10**7)
        ) * t  ## 2pi/(seconds in a year) * seconds passed from integration start
        if (2 * np.pi / (3.154 * 10**7)) * t > 2 * np.pi:
            true_anomaly_earth_rad = true_anomaly_earth_rad - (2 * np.pi)
        mu_sun = 1.327 * 10**11
        c = 2.99792 * 10**5  ## km/s
        earth_state = orbitalElementsToCartesian(
            [1.49597 * 10**8, 0, 0, 0, 0, true_anomaly_earth_rad],
            1.989 * 10**30,
            mu=mu_sun,
        )
        r_s = np.array(earth_state[:3])
        rdot_s = np.array(earth_state[3:])
        r_s_norm = np.linalg.norm(r_s)
        ## For the lense-thirring effect we require earths specific rotational angular momentum
        ## Since we are in a non rotating geocentric frame, this vector just points in the positive z direction
        ## For some reason this magnitude is only given in kg m^2/s, so divide by mass
        earth_angular = np.array([0, 0, (7.05 * 10**33) / (5.9722 * 10**24 * 10**6)])

        ## In the paper beta, gamma "are PPN parameters equal to 1 in GR" this cant be correct because beta is
        ## Nearly zero in GR...
        ## I'll calculate it anyways
        beta = v_norm / c
        gamma = 1 / np.sqrt(1 - beta**2)
        accel_vector = np.array([ax, ay, az])
        a_s = 0
        a_d = 0
        a_lt = 0

        if schwarzchild:
            a_s = (mu / (c**2 * r_norm**3)) * (
                (
                    (
                        2 * (beta + gamma) * (mu / r_norm)
                        - np.dot(gamma * v_vector, v_vector)
                    )
                    * r_vector
                )
                + (2 * (1 + gamma) * (np.dot(r_vector, v_vector)) * v_vector)
            )
            # print("a_s ", np.linalg.norm(a_s))
            accel_vector = accel_vector + a_s
            a_s_magarray.append(np.linalg.norm(a_s))
            a_s_array.append(a_s)

        if lensethirring:
            a_lt = (
                (1 + gamma)
                * (mu / (c**2 * r_norm**3))
                * (
                    (
                        (3 / r_norm**2)
                        * (np.cross(r_vector, v_vector))
                        * (np.dot(r_vector, earth_angular))
                    )
                    + (np.cross(v_vector, earth_angular))
                )
            )
            accel_vector = accel_vector + a_lt
            # print("a_lt ", np.linalg.norm(a_lt))
            a_lt_magarray.append(np.linalg.norm(a_lt))
            a_lt_array.append(a_lt)

        if desitter:
            first_cross = np.cross(rdot_s, (-mu_sun / (c**2 * r_s_norm**3)) * r_s)
            second_cross = np.cross(first_cross, v_vector)
            a_d = (1 + (2 * gamma)) * second_cross
            accel_vector = accel_vector + a_d
            # print("a_d ", np.linalg.norm(a_d))
            a_d_magarray.append(np.linalg.norm(a_d))
            a_d_array.append(a_d)

        return [
            state_vector[3],
            state_vector[4],
            state_vector[5],
            accel_vector[0],
            accel_vector[1],
            accel_vector[2],
        ]

    ## set up our integrator and associated variables
    integrator = integrate.ode(twoBodyDifEq)
    integrator.set_integrator(
        "dop853"
    )  # use 8th order RK method - apparently it's really good
    integrator.set_f_params(mu)  # use earth mass
    integrator.set_initial_value(initial_state_vector, 0)
    dt = time_step  # arbitrary, set by user
    state_array = np.array([initial_state_vector])
    time_array = np.array([0])
    i = 1

    while integrator.successful():
        integrator.integrate(integrator.t + dt)
        time_array = np.append(time_array, [integrator.t], axis=0)
        state_array = np.append(state_array, [integrator.y], axis=0)
        ## Just find some way to tell it passed the initial condition
        ## when it starts we will be getting further away from each x, y, z initial condition
        ## at some point (180 degrees later) we will begin to get closer again, and after that we flag when we get further away again
        ## Except that only works when the initial conditions place you in an already stable orbit...
        ## I'll implement this here for now and see what a good answer is
        if oneOrbit:
            if i > 2:
                ## The norm of the difference of the previous state array and the initial should get larger as the orbit begins to get
                # closer again
                if np.linalg.norm(
                    state_array[i - 2, :3] - initial_state_vector[:3]
                ) > np.linalg.norm(state_array[i - 1, :3] - initial_state_vector[:3]):
                    ## If the previous one was getting closer and the current one is getting further, we know we've passed one orbit
                    if np.linalg.norm(
                        state_array[i - 1, :3] - initial_state_vector[:3]
                    ) < np.linalg.norm(state_array[i, :3] - initial_state_vector[:3]):
                        break
        else:
            if i * time_step > timedOrbit:
                break

        i += 1
    if export_time:
        total_array = np.zeros((len(time_array), 7))

        for i in range(len(time_array)):
            total_array[i, 0] = state_array[i, 0]
            total_array[i, 1] = state_array[i, 1]
            total_array[i, 2] = state_array[i, 2]
            total_array[i, 3] = state_array[i, 3]
            total_array[i, 4] = state_array[i, 4]
            total_array[i, 5] = state_array[i, 5]
            total_array[i, 6] = time_array[i]
        return (
            total_array,
            np.array([a_s_array, a_lt_array, a_d_array]),
            np.array([a_s_magarray, a_lt_magarray, a_d_magarray]),
        )
    else:
        return (
            state_array,
            np.array([a_s_array, a_lt_array, a_d_array]),
            np.array([a_s_magarray, a_lt_magarray, a_d_magarray]),
        )


mu_earth = 3.986 * 10**5
a = (32000 + 17180) / 2
e = (32000 - 17180) / (32000 + 17180)
test_vector = [a, e, 0, 0, 0, 0]
cartesian_test_vector = orbitalElementsToCartesian(test_vector, 0, mu=mu_earth)
orbitTest, relativistic_vectors, relativistic_mags = twoBodyPropRelativistic(
    cartesian_test_vector,
    mu_earth,
    schwarzchild=True,
    desitter=True,
    lensethirring=True,
    oneOrbit=True,
    time_step=100,
)


## I'll also establish the plotting routine here
def plotter(
    state_array,
    relativistic_vectors,
    relativistic_mags,
    plotEnergy=False,
    planet_mass=5.972 * 10**24,
    planet_radius=6.371 * 10**3,
    time_step=10,
):
    radius_earth = 6378  # km
    G = 6.67 * 10**-20

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

    # Create a color map and normalize the froth values
    s_norm = Normalize(vmin=relativistic_mags[0].min(), vmax=relativistic_mags[0].max())
    lt_norm = Normalize(
        vmin=relativistic_mags[1].min(), vmax=relativistic_mags[1].max()
    )
    d_norm = Normalize(vmin=relativistic_mags[2].min(), vmax=relativistic_mags[2].max())
    cmap = plt.get_cmap("magma")
    s_colors = cmap(s_norm(relativistic_mags[0]))
    print(s_colors)
    lt_colors = cmap(lt_norm(relativistic_mags[1]))
    d_colors = cmap(d_norm(relativistic_mags[2]))

    # Trajectory
    # Has to be over for loop
    for i in range(len(state_array)):
        ax.scatter(
            state_array[i, 0],
            state_array[i, 1],
            state_array[i, 2],
            color=s_colors[i, :3],
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

    ## Now I'll plot the vector of the final condition
    ax.quiver(
        state_array[-1, 0],
        state_array[-1, 1],
        state_array[-1, 2],
        state_array[-1, 3],
        state_array[-1, 4],
        state_array[-1, 5],
        color="r",
        label="Final State Vector (km / s)",
    )

    ## Set limits + Graph Specs
    graph_limit = np.max(np.abs(state_array[:, :3]))

    ax.set_xlim([-graph_limit, graph_limit])
    ax.set_ylim([-graph_limit, graph_limit])
    ax.set_zlim([-graph_limit, graph_limit])

    ax.set_xlabel(["X (km)"])
    ax.set_ylabel(["Y (km)"])
    ax.set_zlabel(["Z (km)"])

    ax.set_title(["Thrusting Earth Orbit"])
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
        ax_xyz.set_ylabel("Position Magnitude (km)")
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
        ax_xyz.set_ylabel("Position (km)")
        ax_xyz.legend()

        ax_vxyz = ax2[1]
        ax_vxyz.plot(time_array, state_array[:, 3], "w", label="X Velocity")
        ax_vxyz.plot(time_array, state_array[:, 4], "r", label="Y Velocity")
        ax_vxyz.plot(time_array, state_array[:, 5], "b", label="Z Velocity")
        ax_vxyz.set_xlabel("Time (s)")
        ax_vxyz.set_ylabel("Velocity (km / s)")
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
            energy[i] = kinetic + potential
        ax_energy.set_title("Orbital Energy")
        ax_energy.plot(time_array, energy, "w", label="Total E")
        ax_energy.set_xlabel("Time (s)")
        ax_energy.set_ylabel("Energy")
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
        ax_momentumi.set_ylabel("Specific Angular Momentum (km^2 / s)")
        ax_momentumi.legend()

        ax_momentumj = ax4[1]
        ax_momentumj.yaxis.set_major_formatter(formatter)
        ax_momentumj.plot(
            time_array, angular_momentum[:, 1], "w", label="Angular Momentum j"
        )
        ax_momentumj.set_xlabel("Time (s)")
        ax_momentumj.set_ylabel("Specific Angular Momentum (km^2 / s)")
        ax_momentumj.legend()

        ax_momentumk = ax4[2]
        ax_momentumk.yaxis.set_major_formatter(formatter)
        ax_momentumk.plot(
            time_array, angular_momentum[:, 2], "w", label="Angular Momentum k"
        )
        ax_momentumk.set_xlabel("Time (s)")
        ax_momentumk.set_ylabel("Specific Angular Momentum (km^2 / s)")
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
        ax_a.set_ylabel("Semi-major Axis (km)")
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


plotter(orbitTest, relativistic_vectors, relativistic_mags)
plt.show()
