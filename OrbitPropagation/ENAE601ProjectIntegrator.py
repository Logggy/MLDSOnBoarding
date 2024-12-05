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
