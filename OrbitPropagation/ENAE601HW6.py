import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sympy import symbols, diff, lambdify
from orbitPropagator import (
    cartesianToOrbitalElements,
    orbitalElementsToCartesian,
    twoBodyProp,
)


def twoBodyPropJ2(
    cartesian_state_vector,
    mu,
    time_step=10,
    export_time=False,
    oneOrbit=True,
    timedOrbit=0,
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
    G = 6.67 * 10**-20  # N*m^2/kg^2
    m_earth = 5.972 * 10**24  # kg
    initial_state_vector = [x, y, z, vx, vy, vz]
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
    ## Now we propagate
    ## In this solution we know that F = GMm/r^2
    ## Fun fact - I did not know that the norm of the vector is its magnitude (I thought it meant normalization)
    ## Essentially we already know the solutions to Newtons Equations namely - a = -GMr / mag_r^3, where a and r are vectors
    ## So now it's easy - we have the solutions of the original function (x, y, z, vx, vy, vz)
    ## AND their derivatives (vx, vy, vz, ax, ay, az), all we need to do is integrate to solve the ODE

    def twoBodyDifEqJ2(t, state_vector, mu):
        # r_norm = np.linalg.norm(state_vector[:3])
        # ax, ay, az = -(mu * state_vector[:3]) / (r_norm**3)
        ## We'll need to take the negative gradient of U to get the accelerations
        ## Now we'll define the latitude of the satellite
        # Define the symbolic variables
        ## This potential will give us all the accelerations in x y and z
        x = state_vector[0]
        y = state_vector[1]
        z = state_vector[2]
        ax = du_dx_func(x, y, z)
        ay = du_dy_func(x, y, z)
        az = du_dz_func(x, y, z)
        return [state_vector[3], state_vector[4], state_vector[5], ax, ay, az]

    ## set up our integrator and associated variables
    integrator = integrate.ode(twoBodyDifEqJ2)
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
        return total_array
    else:
        return state_array


radius_e = 6378
mu_earth = 3.986 * 10**5
rp = 1000 + radius_e
e = 0.15
a = rp / (1 - e)
orbitalEOrbit = [
    a,
    e,
    (40 * np.pi) / 180,
    (25 * np.pi) / 180,
    (15 * np.pi) / 180,
    (20 * np.pi) / 180,
]
cartesianOrbit = orbitalElementsToCartesian(orbitalEOrbit, 0, mu=mu_earth)
time_step = 10
j2state = twoBodyPropJ2(
    cartesianOrbit,
    mu_earth,
    oneOrbit=False,
    timedOrbit=3 * 24 * 3600,
    time_step=time_step,
)


def plotter(
    state_array,
    title,
    plotEnergy=False,
    planet_mass=5.972 * 10**24,
    planet_radius=6.378 * 10**6,
    time_step=10,
    mu=3.986 * 10**5,
):
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(10, 10))
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(False)
    # formatter.set_powerlimits((-1, 1))

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
        label="Initial Condition Vector (km / s)",
    )

    ## Set limits + Graph Specs
    graph_limit = np.max(np.abs(state_array[:, :3]))

    ax.set_xlim([-graph_limit, graph_limit])
    ax.set_ylim([-graph_limit, graph_limit])
    ax.set_zlim([-graph_limit, graph_limit])

    ax.set_xlabel(["X (km)"])
    ax.set_ylabel(["Y (km)"])
    ax.set_zlabel(["Z (km)"])

    ax.set_title(title)
    plt.legend()
    time_array = time_step * np.arange(len(state_array))
    ## Now that we've got the 3d plot, we can also add the x, y, z, vx, vy, vz 2d plots
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
        energy = np.zeros(len(state_array))
        for i in range(len(state_array)):
            r = np.linalg.norm(state_array[i, :3])
            r0 = np.linalg.norm(state_array[0, :3])
            j2 = 0.00108248
            radius_e = 6378
            u = (mu / r) * (
                1
                - j2
                * ((radius_e / r) ** 2)
                * ((1.5 * (state_array[i, 2] / r) ** 2) - 0.5)
            )
            u0 = (mu / r0) * (
                1
                - j2
                * ((radius_e / r0) ** 2)
                * ((1.5 * (state_array[0, 2] / r0) ** 2) - 0.5)
            )
            v = np.linalg.norm(state_array[i, 3:])
            v0 = np.linalg.norm(state_array[0, 3:])
            energy0 = (v0**2 * 0.5) - u0
            energyi = (v**2 * 0.5) - u
            energy[i] = energyi - energy0
        ax_energy.yaxis.set_major_formatter(formatter)
        ax_energy.set_title("Time Varying Deviation in Orbital Energy including J2")
        ax_energy.plot(time_array, energy, "w", label="Total E")
        ax_energy.set_xlabel("Time (s)")
        ax_energy.set_ylabel("Energy - Initial Energy (kJ / kg)")
        ax_energy.legend()
        fig4, ax4 = plt.subplots(figsize=(10, 6))  # Single plot
        angular_momentum = np.zeros((len(state_array), 3))
        angular_momentum_mag = np.zeros(len(state_array))
        orbitalEArrayj2 = np.zeros((len(state_array), 6))
        for i in range(len(state_array)):
            orbitalEArrayj2[i] = cartesianToOrbitalElements(state_array[i], mu_earth)
        for i in range(len(angular_momentum)):
            h0 = np.cross(state_array[0, :3], state_array[0, 3:])
            hi = np.cross(state_array[i, :3], state_array[i, 3:])
            h = hi - h0
            angular_momentum[i, 0] = h[0]
            angular_momentum[i, 1] = h[1]
            angular_momentum[i, 2] = h[2]
            angular_momentum_mag[i] = np.sqrt(hi[0] ** 2 + hi[1] ** 2 + hi[2] ** 2)
        ax_momentumi = ax4
        ax_momentumi.yaxis.set_major_formatter(formatter)
        ax_momentumi.set_title("Variation in Angular Momentum in Three Dimensions")
        ax_momentumi.plot(
            time_array, angular_momentum[:, 0], "b", label="Angular Momentum i"
        )
        ax_momentumi.set_xlabel("Time (s)")
        ax_momentumi.set_ylabel("Specific Angular Momentum (km^2 / s)")
        ax_momentumi.legend()

        ax_momentumj = ax4
        ax_momentumj.yaxis.set_major_formatter(formatter)
        ax_momentumj.plot(
            time_array, angular_momentum[:, 1], "r", label="Angular Momentum j"
        )
        ax_momentumj.set_xlabel("Time (s)")
        ax_momentumj.set_ylabel("Specific Angular Momentum (km^2 / s)")
        ax_momentumj.legend()

        ax_momentumk = ax4
        ax_momentumk.yaxis.set_major_formatter(formatter)
        ax_momentumk.plot(
            time_array, angular_momentum[:, 2], "orange", label="Angular Momentum k"
        )
        ax_momentumk.set_xlabel("Time (s)")
        ax_momentumk.set_ylabel("Specific Angular Momentum (km^2 / s)")
        ax_momentumk.legend()
        ax_momentumk.grid()
        fig5, ax5 = plt.subplots(6, 1, figsize=(30, 30))
        ## Now the osculating orbital elements
        ## Now we also have to plot the osculating orbital elements
        ## Now that we have the orbital elements at each time step for time_step = 100s
        ## We need the analytical soltuions for LAN and arg_peri

        radius_e = 6378
        j2 = 0.00108248
        seculan = np.zeros(len(state_array))
        secuperi = np.zeros(len(state_array))
        seculan[0] = (25 * np.pi) / 180
        secuperi[0] = (15 * np.pi) / 180
        for i in range(len(state_array) - 1):
            n = np.sqrt(mu_earth / (orbitalEArrayj2[i, 0] ** 3))
            r_norm = np.linalg.norm(j2state[i, :3])
            p = r_norm * (1 + (orbitalEArrayj2[i, 1] * np.cos(orbitalEArrayj2[i, 5])))
            inc = orbitalEArrayj2[i, 2]
            seculan[i + 1] = seculan[i] + time_step * (
                -3 * n * radius_e**2 * j2 * np.cos(inc)
            ) / (2 * p**2)
            secuperi[i + 1] = secuperi[i] + time_step * (
                3 * n * radius_e**2 * j2 * (4 - (5 * (np.sin(inc) ** 2)))
            ) / (4 * p**2)
        print("Secular Peri: ", secuperi[-1])
        a = np.zeros(len(state_array))
        e = np.zeros(len(state_array))
        inc = np.zeros(len(state_array))
        Omega = np.zeros(len(state_array))
        omega = np.zeros(len(state_array))
        nu = np.zeros(len(state_array))
        for i in range(len(state_array)):
            orbital_state_osculating = cartesianToOrbitalElements(state_array[i], mu)
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
        ax_Omega.plot(time_array, Omega, "w", label="LAN Propagated")
        ax_Omega.plot(time_array, seculan, "r", label="LAN Analytical")

        ax_Omega.set_xlabel("Time (s)")
        ax_Omega.set_ylabel("LAN (rad)")
        ax_Omega.legend()

        ax_omega = ax5[4]
        ax_omega.plot(time_array, omega, "w", label="arg_peri Propagated")
        ax_omega.plot(time_array, secuperi, "r", label="arg_peri Analytical")
        ax_omega.set_xlabel("Time (s)")
        ax_omega.set_ylabel("Arg_Peri (rad)")
        ax_omega.legend()

        ax_nu = ax5[5]
        ax_nu.plot(time_array, nu, "w", label="anomaly")
        ax_nu.set_xlabel("Time (s)")
        ax_nu.set_ylabel("true Anomaly (rad)")
        ax_nu.legend()


## This plots the osculating elements, energy and the angular momentum
plotter(
    j2state,
    ["Stable Earth Orbit w/ J2"],
    planet_radius=6378,
    plotEnergy=True,
    time_step=time_step,
)
plt.show()
