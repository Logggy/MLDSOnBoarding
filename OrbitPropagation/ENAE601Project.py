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


## First I just need to establish a propagator that includes gravitational precession
def calculateRelativity(
    r_vector,
    v_vector,
    mu,
    t,
    schwarzchild=True,
    lensethirring=True,
    desitter=True,
    exaggeration=1,
    planetJ=980,  ## In supporting papers this is the angular momentum for earth they used
    perigee=56.1 * np.pi / 180,
    raan=52.5 * np.pi / 180,
    inc=50 * np.pi / 180,
    planet_inc=23.4 * np.pi / 180,
):
    mu_earth = 3.986 * 10**5
    r_norm = np.linalg.norm(r_vector)
    ## We only need to add a few more terms here: mu_sun, R_s and Rdot_s
    ## R_s = position of the earth with respect to the sun
    ## Rdot_s = velocity of the earth relative to the sun
    ## To do this I will manually evaluate the change in earths true anomaly assuming a circular orbit
    ## Earths true anomaly starts at zero and we know we change 2pi radians per 365 days
    ## Assuming earth starts at some arbitrary true anomaly zero:
    true_anomaly_earth_rad = (
        1.99 * 10**-7
    ) * t  ## 2pi/(seconds in a year) * seconds passed from integration start
    if true_anomaly_earth_rad * t > 2 * np.pi:
        true_anomaly_earth_rad = true_anomaly_earth_rad - (2 * np.pi)
    mu_sun = 1.327 * 10**11
    c = 2.99792 * 10**5  ## km/s
    earth_state = orbitalElementsToCartesian(
        [1.49598 * 10**8, 0.01671, 0, 0, 0, true_anomaly_earth_rad + np.pi],
        1.989 * 10**30,
        mu=mu_sun,
    )
    r_s = np.array(earth_state[:3])
    rdot_s = np.array(earth_state[3:])
    r_s_norm = np.linalg.norm(r_s)
    ## For the lense-thirring effect we require earths specific rotational angular momentum
    ## Since we are in a non rotating geocentric frame, this vector just points in the positive z direction
    ## In the 2010 conventions paper J = 9.8 * 10^8 m^2/s
    earth_angular = np.array([0, 0, planetJ])
    ## In the paper beta, gamma are PPN parameters equal to 1 in GR
    beta = 1  ## PPN parameters
    gamma = 1  ## PPN parameters
    a_s = 0
    a_d = 0
    a_lt = 0

    ## Following terms as provided in Sosnica et al 2021
    if schwarzchild:
        a_s = (mu / (c**2 * r_norm**3)) * (
            (
                ((2 * (beta + gamma) * (mu / r_norm)) - gamma * (v_vector @ v_vector))
                * r_vector
            )
            + (2 * (1 + gamma) * np.dot(r_vector, v_vector) * v_vector)
        )

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
    if desitter:
        first_cross = np.cross(rdot_s, (-mu_sun / (c**2 * r_s_norm**3)) * r_s)
        ## Now the earth is inclined an extra 23.4 degrees, causing our orbit to appear like it
        ## Is inclined an extra 23.4 degrees due to this effect
        ## We will simply apply an R_x rotation to achieve this
        state_vector_nominal = [
            r_vector[0],
            r_vector[1],
            r_vector[2],
            v_vector[0],
            v_vector[1],
            v_vector[2],
        ]
        v_vector_helio = cartesianToOrbitalElements(state_vector_nominal, mu_earth)
        v_vector_helio[2] = v_vector_helio[2] + planet_inc
        v_vector_helio = orbitalElementsToCartesian(v_vector_helio, 0, mu=mu_earth)
        ## By doing this we have created a frame equivalent to the original orbit being inclined to 50 degrees
        ## but with the earth inclined 23.4 degrees compared to the sun
        ## To accomplish this I did it the foolproof way, I rotated back to the base frame (the perifocal frame)
        ## Then rotated it back to the original orbit at inc 50
        second_cross = np.cross((1 + (2 * gamma)) * first_cross, v_vector_helio[3:])
        R_x_sun = np.array(
            [
                [1, 0, 0],
                [0, np.cos(inc + planet_inc), -np.sin(inc + planet_inc)],
                [0, np.sin(inc + planet_inc), np.cos(inc + planet_inc)],
            ]
        )
        R_x_inertial = np.array(
            [
                [1, 0, 0],
                [0, np.cos(inc), -np.sin(inc)],
                [0, np.sin(inc), np.cos(inc)],
            ]
        )
        R_z_periapsis = np.array(
            [
                [np.cos(perigee), -np.sin(perigee), 0],
                [np.sin(perigee), np.cos(perigee), 0],
                [0, 0, 1],
            ]
        )
        R_z = np.array(
            [
                [np.cos(raan), -np.sin(raan), 0],
                [np.sin(raan), np.cos(raan), 0],
                [0, 0, 1],
            ]
        )
        R_tot_sun = R_z @ R_x_sun @ R_z_periapsis

        ##Then rotate back to the true frame
        R_tot_inertial = R_z.T @ R_x_inertial.T @ R_z_periapsis.T

        a_d = np.dot(np.dot(second_cross, R_tot_sun), R_tot_inertial)

    return a_s * exaggeration, a_lt * exaggeration, a_d * exaggeration


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
    J_2=False,
    exaggeration=1,
    planetJ=980,
):
    ## Establish State
    x = cartesian_state_vector[0]
    y = cartesian_state_vector[1]
    z = cartesian_state_vector[2]
    vx = cartesian_state_vector[3]
    vy = cartesian_state_vector[4]
    vz = cartesian_state_vector[5]
    ## Lets establish some constants
    initial_state_vector = [x, y, z, vx, vy, vz]
    ## In case we need J2
    x, y, z = symbols("x y z")
    r = (x**2 + y**2 + z**2) ** (1 / 2)
    j2 = 0.00108248
    radius_e = 6378
    u = (mu / r) * (1 - j2 * ((radius_e / r) ** 2) * ((1.5 * (z / r) ** 2) - 0.5))

    du_dx = diff(u, x)
    du_dy = diff(u, y)
    du_dz = diff(u, z)
    # Convert derivatives to fast numerical functions
    du_dx_func = lambdify((x, y, z), du_dx, "numpy")
    du_dy_func = lambdify((x, y, z), du_dy, "numpy")
    du_dz_func = lambdify((x, y, z), du_dz, "numpy")

    def twoBodyDifEqJ2Relativity(t, state_vector, mu):
        r_vector = state_vector[:3]
        v_vector = state_vector[3:]
        x = state_vector[0]
        y = state_vector[1]
        z = state_vector[2]
        if J_2:
            ax = du_dx_func(x, y, z)
            ay = du_dy_func(x, y, z)
            az = du_dz_func(x, y, z)
        else:  ## If no J2 just the standard newtonian 2 body orbit
            r_norm = np.linalg.norm(state_vector[:3])
            ax, ay, az = -(mu * state_vector[:3]) / (r_norm**3)

        ## Fortunately the magnitudes of these three relativistic accelerations are well described in
        ## Sosnica et al 2021, thank you!
        a_s, a_lt, a_d = calculateRelativity(
            r_vector,
            v_vector,
            mu,
            t,
            schwarzchild,
            lensethirring,
            desitter,
            exaggeration=exaggeration,
            planetJ=planetJ,
        )
        accel_vector = [ax, ay, az]
        if schwarzchild:
            accel_vector = accel_vector + a_s
        if lensethirring:
            accel_vector = accel_vector + a_lt
        if desitter:
            accel_vector = accel_vector + a_d
        return [
            state_vector[3],
            state_vector[4],
            state_vector[5],
            accel_vector[0],
            accel_vector[1],
            accel_vector[2],
        ]

    ## set up our integrator and associated variables
    integrator = integrate.ode(twoBodyDifEqJ2Relativity)
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
        return total_array
    else:
        return state_array


time_step = 50
mu_earth = 3.986 * 10**5
radius_earth = 6371
r_p = 17081 + radius_earth
r_a = 26116 + radius_earth
a = (r_p + r_a) / 2
e = (r_a - r_p) / (r_p + r_a)
inc = 50.15 * np.pi / 180  ## Stated in Sosnica et al 2021

## RAAN and Perigee are not explicitly stated in the paper, and matter based on the perturbations you use
## Later in this code I confirm that I am calculating the forces properly,
## but I have no way to know for certain what RAAN and Periapsis they used

raan = 52.5 * np.pi / 180
perigee = 56.1 * np.pi / 180
test_vector = [a, e, inc, raan, perigee, 0]
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
relativistic_mags = np.zeros((3, len(orbitTest)))
relativistic_vectors = np.zeros((3, len(orbitTest), 3))
orbitTest = np.array(orbitTest)
for i in range(len(orbitTest)):
    a_s, a_lt, a_d = calculateRelativity(
        orbitTest[i, :3], orbitTest[i, 3:], mu_earth, i * time_step
    )
    relativistic_vectors[0, i] = a_s
    relativistic_vectors[1, i] = a_lt
    relativistic_vectors[2, i] = a_d

    relativistic_mags[0, i] = np.linalg.norm(a_s)
    relativistic_mags[1, i] = np.linalg.norm(a_lt)
    relativistic_mags[2, i] = np.linalg.norm(a_d)

print(
    "Schwarz: ",
    "Max: ",
    max(relativistic_mags[0]),
    "Min: ",
    min(relativistic_mags[0]),
    "Med: ",
    np.median(relativistic_mags[0]),
)
print(
    "Lense Thirring: ",
    "Max: ",
    max(relativistic_mags[1]),
    "Min: ",
    min(relativistic_mags[1]),
    "Med: ",
    np.median(relativistic_mags[1]),
)
print(
    "Desitter: ",
    "Max: ",
    max(relativistic_mags[2]),
    "Min: ",
    min(relativistic_mags[2]),
    "Med: ",
    np.median(relativistic_mags[2]),
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
        "ro",
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

    ax.set_title(["E14 Orbit w/ Schwarzchild Term Magnitude"])
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

## Now I'll try to recreate the plots fro fig 1, fig 2, and fig 3
## Start with shcwarzchild
schwarzchild_vectors = np.array(relativistic_vectors[0])
lense_thirring_vectors = np.array(relativistic_vectors[1])
desitter_vectors = np.array(relativistic_vectors[2])

schwarzchild_mags = relativistic_mags[0]
lense_thirring_mags = relativistic_mags[1]
desitter_mags = relativistic_mags[2]

schwarz_min = np.min(schwarzchild_mags)
schwarz_max = np.max(schwarzchild_mags)

lt_min = np.min(lense_thirring_mags)
lt_max = np.max(lense_thirring_mags)

desitter_min = np.min(desitter_mags)
desitter_max = np.max(desitter_mags)

r_vecs = orbitTest[:, :3]

## Rotate each to be 2d
# Define rotation matrix about the x-axis
earth_inc = 23.4 * np.pi / 180
R_z = np.array(
    [[np.cos(raan), -np.sin(raan), 0], [np.sin(raan), np.cos(raan), 0], [0, 0, 1]]
)
R_x = np.array(
    [[1, 0, 0], [0, np.cos(inc), -np.sin(inc)], [0, np.sin(inc), np.cos(inc)]]
)
R_x_sun = np.array(
    [
        [1, 0, 0],
        [0, np.cos(inc + earth_inc), -np.sin(inc + earth_inc)],
        [0, np.sin(inc + earth_inc), np.cos(inc + earth_inc)],
    ]
)
R_z_periapsis = np.array(
    [
        [np.cos(perigee), -np.sin(perigee), 0],
        [np.sin(perigee), np.cos(perigee), 0],
        [0, 0, 1],
    ]
)
R_tot = R_z @ R_x @ R_z_periapsis
R_tot_sun = R_z @ R_x_sun @ R_z_periapsis
# Rotate position vectors (r_vecs) and force vectors (schwarzchild_vectors, etc.)
for i in range(len(orbitTest)):
    r_vecs[i] = np.dot(
        r_vecs[i], R_tot
    )  # Project onto xy-plane (take first two columns)
    schwarzchild_vectors[i] = np.dot(schwarzchild_vectors[i], R_tot)
    ## The lesne thirring force calulcated at the different inclined orbit in the frame I have posed
    ## is equivalent to the earth being inclined naturally with the original inclination of 50
    lense_thirring_vectors[i] = np.dot(lense_thirring_vectors[i], R_tot)
    desitter_vectors[i] = np.dot(desitter_vectors[i], R_tot)

r_vecs = r_vecs[:, :2]
schwarzchild_vectors = schwarzchild_vectors[:, :2]
lense_thirring_vectors = lense_thirring_vectors[:, :2]
desitter_vectors = desitter_vectors[:, :2]


## Start with shcwarzchild
# Function to plot a single force
def plot_force_2d(r_vecs_2d, vectors_2d, mags, title, cmap="viridis"):
    plt.style.use("default")  # Light background with gridlines
    plt.figure(figsize=(8, 8))

    # Normalize the color range explicitly
    norm = Normalize(vmin=np.min(mags), vmax=np.max(mags))
    colors = plt.cm.get_cmap(cmap)(norm(mags))

    # Scatter plot of positions, color-coded by force magnitude
    scatter = plt.scatter(
        r_vecs_2d[:, 0],
        r_vecs_2d[:, 1],
        c=mags,
        cmap=cmap,
        norm=norm,  # Explicitly use the defined normalization
        s=10,
        alpha=0.7,
    )

    # Add vectors as quivers
    scale = 1
    if title == "Schwarzschild Force":
        scale = 10**13
    if title == "Lense-Thirring Force":
        scale = 2 * 10**15
    if title == "de Sitter Force":
        scale = 3 * 10**14
    plt.quiver(
        r_vecs_2d[::3, 0],
        r_vecs_2d[::3, 1],
        vectors_2d[::3, 0] * scale,
        vectors_2d[::3, 1] * scale,
        color=colors[::3],
        scale=50,
        width=0.002,
    )

    # Add a colorbar with scientific notation
    cbar = plt.colorbar(scatter, label="Force Magnitude (km/s^2)")
    cbar.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    cbar.ax.yaxis.get_offset_text().set_fontsize(
        10
    )  # Adjust size of the exponent label

    # Set titles and labels
    # Set plot limits
    plt.xlim(-40000, 40000)
    plt.ylim(-30000, 30000)
    plt.title(title)
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.grid(
        color="gray", linestyle="--", linewidth=0.5, alpha=0.5
    )  # Light gray gridlines
    plt.gca().set_facecolor("white")  # Explicitly set white background
    plt.show()


# Plot Schwarzschild force
plot_force_2d(r_vecs, schwarzchild_vectors, schwarzchild_mags, "Schwarzschild Force")

# Plot Lense-Thirring force
plot_force_2d(
    r_vecs, lense_thirring_vectors, lense_thirring_mags, "Lense-Thirring Force"
)

# Plot de Sitter force
plot_force_2d(r_vecs, desitter_vectors, desitter_mags, "de Sitter Force")


## I am going to recreate the magnitude of the forces for varying circular orbits

# Heights for orbits (in km)
geo_orbit = 35786
gps_orbit = 20184
lageos_orbit = 5850
jason_orbit = 1335
champ_orbit = 350
heights = np.linspace(100, 45000, 1000)

relativities = np.zeros((len(heights), 3))

for i in range(len(heights)):
    vector = [heights[i] + radius_earth, 0, 0, 0, 0, 0]
    cartesian_vector = orbitalElementsToCartesian(vector, 0, mu=mu_earth)
    r = np.array(cartesian_vector[:3])
    v = np.array(cartesian_vector[3:])

    a_s, a_lt, a_d = calculateRelativity(r, v, mu_earth, 0)

    relativities[i, 0] = np.linalg.norm(a_s)
    relativities[i, 1] = np.linalg.norm(a_lt)
    relativities[i, 2] = np.linalg.norm(a_d)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(heights, relativities[:, 0], label="Schwarzschild term ($a_s$)")
plt.plot(heights, relativities[:, 1], label="Lense-Thirring term ($a_{lt}$)")
plt.plot(heights, relativities[:, 2], label="de Sitter term ($a_d$)")

# Adding vertical lines
orbits = [geo_orbit, gps_orbit, lageos_orbit, jason_orbit, champ_orbit]
orbit_labels = ["GEO orbit", "GPS orbit", "LAGEOS orbit", "Jason orbit", "CHAMP orbit"]

for orbit, label in zip(orbits, orbit_labels):
    plt.axvline(x=orbit, color="gray", linestyle="--", label=f"{label} ({orbit} km)")

# Labels and legend
plt.xlabel("Orbit Height (km)")
plt.ylabel("Acceleration (km/s^2)")
plt.yscale("log")  # Assuming forces are better visualized on a log scale
plt.title("Relativistic Force Magnitudes for Circular Orbits With Varying Heights")
plt.legend()
plt.grid(True)
plt.show()


## Lets check the change in the semimajor axis
time_step = 50
aGR = 27978.028
eGR = 0.1612
incGR = 50.15 * np.pi / 180
raanGR = 40 * np.pi / 180
perigeeGR = 0 * np.pi / 180
test_vector_new = [a, e, incGR, raanGR, perigeeGR, np.pi]
test_vectorGR = [
    aGR,
    eGR,
    incGR,
    raanGR,
    perigeeGR,
    np.pi,
]  ## Use the osculating parameters that take into account the GR offsets
cartesian_semimajor_axis = orbitalElementsToCartesian(test_vectorGR, 0, mu=mu_earth)
orbitSemimajor = twoBodyPropRelativistic(
    cartesian_semimajor_axis,
    mu_earth,
    schwarzchild=True,
    desitter=False,
    lensethirring=False,
    oneOrbit=False,
    timedOrbit=100000,
    time_step=time_step,
    export_time=False,
)

semimajor_change = np.zeros(len(orbitSemimajor))
eccentric_change = np.zeros(len(orbitSemimajor))
for i in range(len(orbitSemimajor)):
    semimajor_change[i] = (
        -test_vectorGR[0] + cartesianToOrbitalElements(orbitSemimajor[i], mu_earth)[0]
    )
    eccentric_change[i] = (
        -test_vectorGR[1] + cartesianToOrbitalElements(orbitSemimajor[i], mu_earth)[1]
    )
time = np.linspace(0, len(orbitSemimajor) * time_step, len(orbitSemimajor))
plt.plot(time, semimajor_change)
plt.title("Osculating a for E14")
plt.show()
plt.plot(time, eccentric_change)
plt.title("Osculating e for E14")
plt.show()
# We must now recreate figure 9 from the Sosnica 2021
# To do so we: Use only the schwarzchild term for our orbit, and we
# Multiply the difference between the nominal r and the new one by 4*10**8
# Nominal Orbit
cartesian_nominal_vector = orbitalElementsToCartesian(test_vector_new, 0, mu=mu_earth)
orbitNominal = twoBodyPropRelativistic(
    cartesian_nominal_vector,
    mu_earth,
    schwarzchild=False,
    desitter=False,
    lensethirring=False,
    oneOrbit=True,
    time_step=time_step,
    export_time=False,
)

## Exaggerated
cartesian_exaggeratedSC_vector = orbitalElementsToCartesian(
    test_vectorGR, 0, mu=mu_earth
)
orbitExaggeratedSC = twoBodyPropRelativistic(
    cartesian_exaggeratedSC_vector,
    mu_earth,
    schwarzchild=True,
    desitter=False,
    lensethirring=False,
    oneOrbit=True,
    time_step=time_step,
    export_time=False,
)
exaggerated = np.zeros((len(orbitExaggeratedSC), 3))
for i in range(len(orbitExaggeratedSC)):
    orbitOE = cartesianToOrbitalElements(orbitExaggeratedSC[i], mu_earth)
    eccent = orbitOE[1] - (4 * 10**8 * (test_vectorGR[1] - orbitOE[1]))
    if eccent < 0:
        eccent = 0
    semi = orbitOE[0] - (4 * 10**8 * (test_vectorGR[0] - orbitOE[0]))
    print(semi, eccent)
    exaggerated[i] = orbitalElementsToCartesian(
        [
            semi,
            eccent,
            test_vectorGR[2],
            test_vectorGR[3],
            test_vectorGR[4],
            orbitOE[5],
        ],
        0,
        mu_earth,
    )[:3]


# Create the figure
## Rotate each to be 2d
# Define rotation matrix about the x-axis
R_z = np.array(
    [
        [np.cos(raanGR), -np.sin(raanGR), 0],
        [np.sin(raanGR), np.cos(raanGR), 0],
        [0, 0, 1],
    ]
)
R_x = np.array(
    [[1, 0, 0], [0, np.cos(incGR), -np.sin(incGR)], [0, np.sin(incGR), np.cos(incGR)]]
)
R_z_periapsis = np.array(
    [
        [np.cos(perigeeGR), -np.sin(perigeeGR), 0],
        [np.sin(perigeeGR), np.cos(perigeeGR), 0],
        [0, 0, 1],
    ]
)
R_tot = R_z @ R_x @ R_z_periapsis
# Rotate position vectors (r_vecs) and force vectors (schwarzchild_vectors, etc.)
orbitNominal = orbitNominal[:, :3]
exaggerated = exaggerated[:, :3]
for i in range(len(orbitNominal)):
    orbitNominal[i] = np.dot(
        orbitNominal[i], R_tot
    )  # Project onto xy-plane (take first two columns)
    exaggerated[i] = np.dot(
        exaggerated[i], R_tot
    )  # Project onto xy-plane (take first two columns)

# Create the 2D plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot the nominal orbit
ax.plot(
    orbitNominal[:, 0],  # X-axis
    orbitNominal[:, 1],  # Y-axis
    label="Nominal Orbit",
    color="green",
)

# Plot the exaggerated orbit
ax.plot(
    exaggerated[:, 0],  # X-axis
    exaggerated[:, 1],  # Y-axis
    label="Exaggerated Orbit",
    color="blue",
)

# Add labels, title, and legend
ax.set_xlabel("X [km]")
ax.set_ylabel("Y [km]")
ax.set_title("Nominal vs Exaggerated by 4*10^8 Displacement for E14")
ax.legend()
ax.grid()

# Show the plot
plt.show()
## Please note - the initial exaggerated offset in the orbit is incorrect! This is because they never explicitly give
## the true nominal value they used in the paper and I would have to offset the first point manually
## when exagerrating the orbit!
## This is good enough for now, Ill discuss in my findings

## So far we have exlpored semimajor axis changes on circular orbits, lets explore different eccentricities at
## a 10000 km circular orbit height, zero inclination
# Semi-major axis (a) for 10,000 km orbit
semi_major_axis = 10000 + radius_earth  # in km

# Eccentricity range from 0 to 0.9
eccentricities = np.linspace(0, 0.9, 100)

relativities = np.zeros((len(eccentricities), 3))

for i in range(len(eccentricities)):
    state = orbitalElementsToCartesian(
        [semi_major_axis, eccentricities[i], 0, 0, 0, 0], 0, mu_earth
    )
    r_periapsis = np.array(state[:3])
    v_periapsis = np.array(state[3:])
    a_s, a_lt, a_d = calculateRelativity(r_periapsis, v_periapsis, mu_earth, 0)
    relativities[i, 0] = np.linalg.norm(a_s)
    relativities[i, 1] = np.linalg.norm(a_lt)
    relativities[i, 2] = np.linalg.norm(a_d)


plt.figure(figsize=(10, 6))
plt.plot(eccentricities, relativities[:, 0], label="Schwarzschild term ($a_s$)")
plt.plot(eccentricities, relativities[:, 1], label="Lense-Thirring term ($a_{lt}$)")
plt.plot(eccentricities, relativities[:, 2], label="de Sitter term ($a_d$)")


plt.xlabel("Orbit Eccentricity")
plt.ylabel("Acceleration (km/s^2)")
plt.yscale("log")
plt.title("Relativistic Force Magnitudes for 10,000 km Orbit vs. Eccentricity")
plt.legend()
plt.grid(True)
plt.show()

## Now lets see about inclination!
semi_major_axis = 10000 + radius_earth  # in km

# Inclinations from -90 to 90
incs = np.linspace(-np.pi / 2, np.pi / 2, 100)

relativities = np.zeros((len(incs), 3))

for i in range(len(incs)):
    state = orbitalElementsToCartesian(
        [semi_major_axis, 0, incs[i], 0, 0, 0], 0, mu_earth
    )
    r_periapsis = np.array(state[:3])
    v_periapsis = np.array(state[3:])
    a_s, a_lt, a_d = calculateRelativity(r_periapsis, v_periapsis, mu_earth, 0)
    relativities[i, 0] = np.linalg.norm(a_s)
    relativities[i, 1] = np.linalg.norm(a_lt)
    relativities[i, 2] = np.linalg.norm(a_d)


plt.figure(figsize=(10, 6))
plt.plot(incs, relativities[:, 0], label="Schwarzschild term ($a_s$)")
plt.plot(incs, relativities[:, 1], label="Lense-Thirring term ($a_{lt}$)")
plt.plot(incs, relativities[:, 2], label="de Sitter term ($a_d$)")


plt.xlabel("Orbit Inclination")
plt.ylabel("Acceleration (km/s^2)")
plt.yscale("log")
plt.title("Relativistic Force Magnitudes for 10,000 km Orbit vs. Inclination")
plt.legend()
plt.grid(True)
plt.show()

## What if earth had a different mass with the same rotation
semi_major_axis = 10000 + radius_earth  # in km

# Earth masses from M/10 to 10M
mus = np.linspace(mu_earth / 10, mu_earth * 10, 1000)

relativities = np.zeros((len(mus), 3))

for i in range(len(mus)):
    state = orbitalElementsToCartesian([semi_major_axis, 0, 0, 0, 0, 0], 0, mus[i])
    r_periapsis = np.array(state[:3])
    v_periapsis = np.array(state[3:])
    a_s, a_lt, a_d = calculateRelativity(r_periapsis, v_periapsis, mus[i], 0)
    relativities[i, 0] = np.linalg.norm(a_s)
    relativities[i, 1] = np.linalg.norm(a_lt)
    relativities[i, 2] = np.linalg.norm(a_d)


plt.figure(figsize=(10, 6))
plt.plot(mus / mu_earth, relativities[:, 0], label="Schwarzschild term ($a_s$)")
plt.plot(mus / mu_earth, relativities[:, 1], label="Lense-Thirring term ($a_{lt}$)")
plt.plot(mus / mu_earth, relativities[:, 2], label="de Sitter term ($a_d$)")


plt.xlabel("Earth Mass (earth masses)")
plt.ylabel("Acceleration (km/s^2)")
plt.yscale("log")
plt.title("Relativistic Force Magnitudes for 10,000 km Orbit vs. Earth Mass")
plt.legend()
plt.grid(True)
plt.show()

## Very cool, how about rotation speed
semi_major_axis = 10000 + radius_earth  # in km

# Earth masses from M/10 to 10M
rot = np.linspace(980 / 100, 980 * 100, 1000)

relativities = np.zeros(((len(rot)), 3))

for i in range(len(rot)):
    state = orbitalElementsToCartesian([semi_major_axis, 0, 0, 0, 0, 0], 0, mu_earth)
    r_periapsis = np.array(state[:3])
    v_periapsis = np.array(state[3:])
    a_s, a_lt, a_d = calculateRelativity(
        r_periapsis, v_periapsis, mu_earth, 0, planetJ=rot[i]
    )
    relativities[i, 0] = np.linalg.norm(a_s)
    relativities[i, 1] = np.linalg.norm(a_lt)
    relativities[i, 2] = np.linalg.norm(a_d)


plt.figure(figsize=(10, 6))
plt.plot(rot / 980, relativities[:, 0], label="Schwarzschild term ($a_s$)")
plt.plot(rot / 980, relativities[:, 1], label="Lense-Thirring term ($a_{lt}$)")
plt.plot(rot / 980, relativities[:, 2], label="de Sitter term ($a_d$)")


plt.xlabel("Fraction of Earth Rotation Speed")
plt.ylabel("Acceleration (km/s^2)")
plt.yscale("log")
plt.title("Relativistic Force Magnitudes for 10,000 km Orbit vs. Earth Rotation")
plt.legend()
plt.grid(True)
plt.show()

## Now lets test against some nominal orbits
## First we'll get the osculating elements for E14 due to all three forces over a single orbit
time_step = 500
test_vector = [a, e, inc, raan, perigee, 0]
cartesian_test_vector = orbitalElementsToCartesian(test_vector, 0, mu=mu_earth)
orbitTest = twoBodyPropRelativistic(
    cartesian_test_vector,
    mu_earth,
    schwarzchild=True,
    desitter=True,
    lensethirring=True,
    oneOrbit=False,
    timedOrbit=60 * 60 * 24 * 2,
    time_step=time_step,
    export_time=False,
)

semimajor_axis = np.zeros(len(orbitTest))
eccentricity = np.zeros(len(orbitTest))
inclination = np.zeros(len(orbitTest))
rightascension = np.zeros(len(orbitTest))
argumentperi = np.zeros(len(orbitTest))
nu = np.zeros(len(orbitTest))
for i in range(len(orbitTest)):
    oetemp = cartesianToOrbitalElements(orbitTest[i], mu_earth)
    semimajor_axis[i] = oetemp[0]
    eccentricity[i] = oetemp[1]
    inclination[i] = oetemp[2]
    rightascension[i] = oetemp[3]
    argumentperi[i] = oetemp[4]
    nu[i] = oetemp[5]
# Time steps
time_steps = np.arange(0, len(orbitTest)) * time_step

# Convert angles to degrees for better readability
inclination_deg = np.rad2deg(inclination)
rightascension_deg = np.rad2deg(rightascension)
argumentperi_deg = np.rad2deg(argumentperi)
true_anomaly_deg = np.rad2deg(nu)

# Create a figure with subplots
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
fig.suptitle("Osculating Orbital Elements over Time", fontsize=16)


# Semi-major axis
axes[0, 0].plot(time_steps, semimajor_axis, color="blue")
axes[0, 0].set_xlabel("Time (s)")
axes[0, 0].set_ylabel("Semi-major Axis (km)")
axes[0, 0].grid(True)

# Eccentricity
axes[0, 1].plot(time_steps, eccentricity, color="orange")
axes[0, 1].set_xlabel("Time (s)")
axes[0, 1].set_ylabel("Eccentricity")
axes[0, 1].grid(True)

# Inclination
axes[1, 0].plot(time_steps, inclination_deg, color="green")
axes[1, 0].set_xlabel("Time (s)")
axes[1, 0].set_ylabel("Inclination (degrees)")
axes[1, 0].grid(True)

# RAAN
axes[1, 1].plot(time_steps, rightascension_deg, color="red")
axes[1, 1].set_xlabel("Time (s)")
axes[1, 1].set_ylabel("RAAN (degrees)")
axes[1, 1].grid(True)

# Argument of Periapsis
axes[2, 0].plot(time_steps, argumentperi_deg, color="purple")
axes[2, 0].set_xlabel("Time (s)")
axes[2, 0].set_ylabel("Argument of Periapsis (degrees)")
axes[2, 0].grid(True)

# True Anomaly
axes[2, 1].plot(time_steps, true_anomaly_deg, color="brown")
axes[2, 1].set_xlabel("Time (s)")
axes[2, 1].set_ylabel("True Anomaly (degrees)")
axes[2, 1].grid(True)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the title
plt.show()

## Now we'll do two demonstrations, first dieplaying the precession of e14 after a long period of time
time_step = 500
test_vector = [a, e, inc, raan, perigee, 0]
cartesian_test_vector = orbitalElementsToCartesian(test_vector, 0, mu=mu_earth)
orbitTest = twoBodyPropRelativistic(
    cartesian_test_vector,
    mu_earth,
    schwarzchild=True,
    desitter=True,
    lensethirring=True,
    oneOrbit=False,
    timedOrbit=60 * 60 * 24 * 30,
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
x_one = orbitOne[:, 0]
y_one = orbitOne[:, 1]
z_one = orbitOne[:, 2]

x_last = orbitLast[:, 0]
y_last = orbitLast[:, 1]
z_last = orbitLast[:, 2]

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Plot the orbits
ax.plot(x_one, y_one, z_one, label="First Orbit", color="blue", linestyle="--")
ax.plot(x_last, y_last, z_last, label="Last Orbit", color="red", linestyle="-")

# Labels and title
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
ax.set_title("First and last orbits of E14 separated by a month")

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

## We have our first orbital comparisons, now lets test some special cases

## Case 1: high mass earth (10x) for E14 Orbit
mu_big = mu_earth * 10
time_step = 500
test_vector = [a, inc, inc, raan, perigee, 0]
cartesian_test_vector = orbitalElementsToCartesian(test_vector, 0, mu=mu_big)
orbitTest = twoBodyPropRelativistic(
    cartesian_test_vector,
    mu_big,
    schwarzchild=True,
    desitter=True,
    lensethirring=True,
    oneOrbit=False,
    timedOrbit=60 * 60 * 24 * 30,
    time_step=time_step,
    export_time=False,
)

# PLot the first and last orbits from orbitTest
first_orbit = orbitTest[0]
last_orbit = orbitTest[-1]

orbitOne = twoBodyPropRelativistic(
    first_orbit,
    mu_big,
    schwarzchild=False,
    desitter=False,
    lensethirring=False,
    oneOrbit=True,
    time_step=50,
    export_time=False,
)

orbitLast = twoBodyPropRelativistic(
    last_orbit,
    mu_big,
    schwarzchild=False,
    desitter=False,
    lensethirring=False,
    oneOrbit=True,
    time_step=50,
    export_time=False,
)

# Extract the x, y, and z coordinates for both orbits
x_one = orbitOne[:, 0]
y_one = orbitOne[:, 1]
z_one = orbitOne[:, 2]

x_last = orbitLast[:, 0]
y_last = orbitLast[:, 1]
z_last = orbitLast[:, 2]

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Plot the orbits
ax.plot(x_one, y_one, z_one, label="First Orbit", color="blue", linestyle="--")
ax.plot(x_last, y_last, z_last, label="Last Orbit", color="red", linestyle="-")

# Labels and title
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
ax.set_title(
    "First and last orbits of E14 around 10x earth mass object separated by a month"
)

# Add a legend
ax.legend()

# Show the plot
plt.show()
first_orbitOE = cartesianToOrbitalElements(first_orbit, mu_big)
last_orbitOE = cartesianToOrbitalElements(last_orbit, mu_big)
print(first_orbitOE)
print(last_orbitOE)
## Set both to periapsis
first_orbitOE[5] = 0
last_orbitOE[5] = 0

## Lets see how far apart they are after thirty days
first_orbitPos = np.array(orbitalElementsToCartesian(first_orbitOE, 0, mu=mu_big))[:3]
last_orbitPos = np.array(orbitalElementsToCartesian(last_orbitOE, 0, mu=mu_big))[:3]

distance = np.linalg.norm(last_orbitPos - first_orbitPos)

## Due to relativity after 30 days, periapsis has moved by
print(
    "Due to relativity after 30 days, periapsis has moved by: ", distance * 10**6, "mm"
)


## What if we were orbiting a pulsar?
## Case 2: pulsar
## Crab nebula weighs 2.8 × 10^30 kg
## angular momentum 2.10x10^34 kg km^2/s
G = 6.674 * 10**-20
mu_big = G * 2.8 * 10**30
time_step = 500
test_vector = [a, inc, inc, raan, perigee, 0]
cartesian_test_vector = orbitalElementsToCartesian(test_vector, 0, mu=mu_big)
print(cartesian_test_vector)
orbitTest = twoBodyPropRelativistic(
    cartesian_test_vector,
    mu_big,
    schwarzchild=True,
    desitter=False,
    lensethirring=True,
    oneOrbit=False,
    timedOrbit=60 * 60 * 24,
    time_step=time_step,
    export_time=False,
    planetJ=(2.10 * 10**34) / (2.8 * 10**30),
)

# PLot the first and last orbits from orbitTest
first_orbit = orbitTest[0]
last_orbit = orbitTest[-1]

orbitOne = twoBodyPropRelativistic(
    first_orbit,
    mu_big,
    schwarzchild=False,
    desitter=False,
    lensethirring=False,
    oneOrbit=True,
    time_step=50,
    export_time=False,
)

orbitLast = twoBodyPropRelativistic(
    last_orbit,
    mu_big,
    schwarzchild=False,
    desitter=False,
    lensethirring=False,
    oneOrbit=True,
    time_step=50,
    export_time=False,
)

# Extract the x, y, and z coordinates for both orbits
x_one = orbitOne[:, 0]
y_one = orbitOne[:, 1]
z_one = orbitOne[:, 2]

x_last = orbitLast[:, 0]
y_last = orbitLast[:, 1]
z_last = orbitLast[:, 2]

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Plot the orbits
ax.plot(x_one, y_one, z_one, label="First Orbit", color="blue", linestyle="--")
ax.plot(x_last, y_last, z_last, label="Last Orbit", color="red", linestyle="-")

# Labels and title
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
ax.set_title(
    "First and last orbits of E14 around Crab Nebula Pulsar orbit separated by a month"
)

# Add a legend
ax.legend()

# Show the plot
plt.show()
first_orbitOE = cartesianToOrbitalElements(first_orbit, mu_big)
last_orbitOE = cartesianToOrbitalElements(last_orbit, mu_big)
print(first_orbitOE)
print(last_orbitOE)
## Set both to periapsis
first_orbitOE[5] = 0
last_orbitOE[5] = 0

## Lets see how far apart they are after thirty days
first_orbitPos = np.array(orbitalElementsToCartesian(first_orbitOE, 0, mu=mu_big))[:3]
last_orbitPos = np.array(orbitalElementsToCartesian(last_orbitOE, 0, mu=mu_big))[:3]

distance = np.linalg.norm(last_orbitPos - first_orbitPos)

## Due to relativity after 30 days, periapsis has moved by
print(
    "Due to relativity after 30 days, periapsis has moved by: ", distance * 10**6, "mm"
)
