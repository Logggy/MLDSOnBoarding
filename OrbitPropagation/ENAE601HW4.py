import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from orbitPropagator import (
    cartesianToOrbitalElements,
    orbitalElementsToCartesian,
    twoBodyProp,
)


## Let's take the same two body propagator that I had before and add in the thrust term
## I will modify it to take in the thrust with units kN/kg
def twoBodyPropWThrust(
    cartesian_state_vector,
    time_step=10,
    export_time=False,
    oneOrbit=False,
    timedOrbit=100,
    bodyMass=5.972 * 10**24,
    thrust_mag=10**-6,
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
    G = 6.67 * 10**-20  # km^3/kgs^2
    initial_state_vector = [x, y, z, vx, vy, vz]

    ## Now we propagate
    ## In this solution we know that F = GMm/r^2
    ## Essentially we already know the solutions to Newtons Equations namely - a = -GMr / mag_r^3, where a and r are vectors
    ## So now it's easy - we have the solutions of the original function (x, y, z, vx, vy, vz)
    ## AND their derivatives (vx, vy, vz, ax, ay, az), all we need to do is integrate to solve the ODE
    ## We are applying an acceleration in the direction of the velocity vector, so we can just add this in within the difeq itself

    def twoBodyDifEq(t, state_vector, M):
        r_norm = np.linalg.norm(state_vector[:3])
        ax, ay, az = -(3.986 * 10**5 * state_vector[:3]) / (r_norm**3)
        # Now we just need to add the thrust in the direction of the velocity
        # So we find the relative magnitudes of each of the velocity vector components
        v_norm = np.linalg.norm(state_vector[3:])

        ## Now add the thrust component to each acceleration
        thrustx = thrust_mag * (state_vector[3] / v_norm)
        thrusty = thrust_mag * (state_vector[4] / v_norm)
        thrustz = thrust_mag * (state_vector[5] / v_norm)

        ax = ax + thrustx
        ay = ay + thrusty
        az = az + thrustz
        return [state_vector[3], state_vector[4], state_vector[5], ax, ay, az]

    ## set up our integrator and associated variables
    integrator = integrate.ode(twoBodyDifEq)
    integrator.set_integrator(
        "dop853"
    )  # use 8th order RK method - apparently it's really good
    integrator.set_f_params(bodyMass)  # use earth mass
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


## I'll also establish the plotting routine here
def plotter(
    state_array,
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

    ax.set_xlim([-5 * 10**5, 5 * 10**5])
    ax.set_ylim([-5 * 10**5, 5 * 10**5])
    ax.set_zlim([-5 * 10**5, 5 * 10**5])

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


# Now we do this for 2t_esc

t_esc = (7.29 / 10**-6) * (1 - (((20 * (10**-6) ** 2 * 7500**2) / 7.29**4) ** (1 / 8)))

initial_state = [7500, 0, 0, 0, 0, 7.29]
dt = 1000
# # for a thrust 10E-6
thrusting_array = twoBodyPropWThrust(initial_state, timedOrbit=2 * t_esc, time_step=dt)
plotter(thrusting_array)
plt.show()
print("The final state of the Spacecraft was: ", thrusting_array[-1])

## Now we just find the timestep where E > 0
planet_mass = 5.972 * 10**24
G = 6.67 * 10**-20
energy = -1
i = 0
while energy < 0:
    ## Energy is Kinetic plus Potential
    kinetic = (
        thrusting_array[i, 3] ** 2
        + thrusting_array[i, 4] ** 2
        + thrusting_array[i, 5] ** 2
    ) * 0.5
    potential = -(planet_mass * G) / np.sqrt(
        thrusting_array[i, 0] ** 2
        + thrusting_array[i, 1] ** 2
        + thrusting_array[i, 2] ** 2
    )
    energy = kinetic + potential
    i += 1

print(
    "The time of escape determined by positive orbital energy is: ", i * dt, " seconds"
)

# The analytical solution underestimates the time of escape. While the derivation we covered in class assumes constant
# thrust (which is fine) it ALSO assume that in the calculation for velocity that the orbit is always almost circular.
# This results in a constant underestimation of the velocity, and an underestimation for r and r_esc, resulting in the ultimate
# underestimation of t_esc

## Now we just need to figure out when the spacecraft reaches 8000 km

i = 0
r_mag = 0
while r_mag < 8000:
    r_mag = np.linalg.norm(thrusting_array[i, :3])
    i += 1

print(
    "The total time it takes for this spacecraft to get to an 8000km radius orbit is: ",
    i * dt,
    " seconds",
)

## Whereas the tof of flight for a Hohmann transfer is just half the period of an orbit with that semimajor axis
## First I have to find the semimajor axis of an orbit with periapsis at 7500km and apoapsis at 8000km
# we can find the semilatus rectum from the radius and velocity at apoapsis

r_a = 8000
r_p = 7500

a = (r_a + r_p) / 2

hohmann_trans_period = 2 * np.pi * np.sqrt(a**3 / (3.986 * 10**5))

print(
    "The time of flight to get to 8000km radius using a Hohmann transfer is: ",
    hohmann_trans_period * 0.5,
    " seconds",
)


# Now I Need to code a lambert solver


def lambertSolve(
    start_planet_state, end_planet_state, dt, shortLong, mu_sun=132712440018
):
    planet1pos = np.array(start_planet_state[:3])
    planet1vel = np.array(start_planet_state[3:])
    planet2pos = np.array(end_planet_state[:3])
    planet2vel = np.array(end_planet_state[3:])
    cosdeltav = np.dot(planet1pos, planet2pos) / (
        np.linalg.norm(planet1pos) * np.linalg.norm(planet2pos)
    )
    # sindeltav = 0
    A = shortLong * np.sqrt(
        (np.linalg.norm(planet1pos) * np.linalg.norm(planet2pos)) * (1 + cosdeltav)
    )

    if A == 0:
        print("This cannot be done lol")
        return

    phi_n = 0
    c2 = 1 / 2
    c3 = 1 / 6
    phi_up = 4 * np.pi**2
    phi_low = -4 * np.pi

    y_n = (
        np.linalg.norm(planet1pos)
        + np.linalg.norm(planet2pos)
        + A * ((phi_n * c3) - 1) / np.sqrt(c2)
    )
    chi_n = np.sqrt(y_n / c2)
    deltat_n = ((chi_n**3 * c3) + (A * np.sqrt(y_n))) / np.sqrt(mu_sun)

    def findcs(phi):
        if phi > 10**-6:
            c2 = (1 - np.cos(np.sqrt(phi))) / phi
            c3 = (np.sqrt(phi) - np.sin(np.sqrt(phi))) / np.sqrt(phi**3)
            return c2, c3
        elif phi < -(10**-6):
            c2 = (1 - np.cosh(np.sqrt(-phi))) / phi
            c3 = (np.sinh(np.sqrt(-phi)) - np.sqrt(-phi)) / np.sqrt((-phi) ** 3)
            return c2, c3
        else:
            return 1 / 2, 1 / 6

    while np.abs(deltat_n - dt) > 10**-6:
        if A > 0 and y_n < 0:
            print("Pick a new phi_low, this one sucks")
        if deltat_n <= dt:
            phi_low = phi_n
        else:
            phi_up = phi_n

        phi_nplus = (phi_up + phi_low) / 2

        c2, c3 = findcs(phi_nplus)

        phi_n = phi_nplus

        y_n = (
            np.linalg.norm(planet1pos)
            + np.linalg.norm(planet2pos)
            + A * ((phi_n * c3) - 1) / np.sqrt(c2)
        )
        chi_n = np.sqrt(y_n / c2)
        deltat_n = ((chi_n**3 * c3) + (A * np.sqrt(y_n))) / np.sqrt(mu_sun)
    f = 1 - (y_n / np.linalg.norm(planet1pos))
    gdot = 1 - (y_n / np.linalg.norm(planet2pos))
    g = A * np.sqrt(y_n / mu_sun)
    initial_velocity = (planet2pos - (f * planet1pos)) / g
    final_velocity = ((gdot * planet2pos) - planet1pos) / g

    return initial_velocity, final_velocity


start_planet_state = [-128169484.29, -190592298.12, -844880.03]
end_planet_state = [483382929.98, -587464623.05, -8381282.40]
start_planet_velocity = [21.02, -11.45, -0.76]
final_planet_velocity = [9.93, 8.92, -0.26]
dt = 830 * 24 * 60 * 60
# Lambert Solver for Short
initial_velocity, final_velocity = lambertSolve(
    start_planet_state, end_planet_state, dt, 1
)

print(
    "For a short transfer from mars to jupiter: Initial Velocity = ",
    initial_velocity,
    " Final Velocity = ",
    final_velocity,
)
# Question 3 - prove that the lambert solver works with numerical integration
testing_state_array = twoBodyProp(
    [
        -128169484.29,
        -190592298.12,
        -844880.03,
        initial_velocity[0],
        initial_velocity[1],
        initial_velocity[2],
    ],
    132712440018,
    time_step=1000,
    oneOrbit=False,
    timedOrbit=830 * 24 * 60 * 60,
)
print("Final Position and Velocity after 830 Day transfer: ", testing_state_array[-1])
print("A difference of: ", final_velocity - testing_state_array[-1, 3:])
# Question 4 - Generate delta v's for a wide range of time of flights from 1 to 3 years

time_steps = 50
times = np.linspace((365 * 60 * 60 * 24), (3 * 365 * 60 * 60 * 24), time_steps)
deltaVshort = np.zeros(time_steps)
deltaVlong = np.zeros(time_steps)

for i in range(time_steps):
    initial_velocity, final_velocity = lambertSolve(
        start_planet_state, end_planet_state, times[i], 1
    )
    deltaVshort[i] = np.abs(
        np.linalg.norm(final_planet_velocity - final_velocity)
        + np.linalg.norm(initial_velocity - start_planet_velocity)
    )

    initial_velocity2, final_velocity2 = lambertSolve(
        start_planet_state, end_planet_state, times[i], -1
    )
    deltaVlong[i] = np.abs(
        np.linalg.norm(final_planet_velocity - final_velocity2)
        + np.linalg.norm(initial_velocity2 - start_planet_velocity)
    )
# to find the hohmann transfer I will have to do the patched conics
# I found it to be 10.20 km/s
hohmann = np.ones(time_steps) * 10.20
plt.plot(times / (365 * 24 * 60 * 60), hohmann, color="blue", label="hohmann")
plt.plot(times / (365 * 24 * 60 * 60), deltaVshort, color="red", label="short")
plt.plot(times / (365 * 24 * 60 * 60), deltaVlong, color="green", label="long")
plt.xlabel("Time of Flight (Years)")
plt.ylabel("Delta-V Required (km / s)")
plt.title("Delta V required to complete transfer vs Time of Flight")
plt.ylim(bottom=0)
plt.grid()
plt.legend()
plt.show()
