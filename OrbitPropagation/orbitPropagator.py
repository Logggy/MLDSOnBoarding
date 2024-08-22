import numpy as np
import scipy.integrate as integrate

## Here we will set up the propagator we will use in this task!

## The goal here is simple - I need to set up a propagator that takes the input of x, y, z, vx, vy, vz
## and outputs an orbit

## First we will make a propagator that assumes 2BD


## Propagator for 2BD, requires cartesian state_array input with desired time step resolution
## This function returns a full state array for each time step
def twoBodyProp(
    x, y, z, vx, vy, vz, time_step=10, export_time=False
):  ## Contemplate adding an argument for example initial conditions!
    ## Lets establish some constants
    G = 6.67 * 10**-11  # N*m^2/kg^2
    m_earth = 5.972 * 10**24  # kg
    initial_state_vector = [x, y, z, vx, vy, vz]

    ## Now we propagate
    ## In this solution we know that F = GMm/r^2
    ## Fun fact - I did not know that the norm of the vector is its magnitude (I thought it meant normalization)
    ## Essentially we already know the solutions to Newtons Equations namely - a = -GMr / mag_r^3, where a and r are vectors
    ## So now it's easy - we have the solutions of the original function (x, y, z, vx, vy, vz)
    ## AND their derivatives (vx, vy, vz, ax, ay, az), all we need to do is integrate to solve the ODE

    def twoBodyDifEq(t, state_vector, M):
        r_norm = np.linalg.norm(state_vector[:3])
        ax, ay, az = -(G * M * state_vector[:3]) / (r_norm**3)

        return [state_vector[3], state_vector[4], state_vector[5], ax, ay, az]

    ## Now we want to propagate for one orbit so I figure I do not want to choose time steps all willy nilly, lets make it smart
    ## My first idea (since this is all ideal 2BD) is to set it to stop when we get back to the starting point by:
    ## 1. set a condition where the propagation stops when the initial x, y, z position is passed (after some time has passed)
    # - failed too sensitive to step size
    ## 2. If you know the eccentricity and the semi-major axis then you can stop when the orbit encompasses the correct area
    # - too complicated probably
    ## 3. rework 1 but how??? Flag the FIRST time slope changes from positive to negative!!! (I thought of this myself!!!!!)
    ## this itself actually isn't enough, you probably want to run a root finder on the velocity curve or something
    # (I'm doing the crude way first)
    ## I wonder what the right answer is lol, its probably just finding the orbital parameters then you don't even have to propagate

    ## set up our integrator and associated variables
    integrator = integrate.ode(twoBodyDifEq)
    integrator.set_integrator(
        "dop853"
    )  # use 8th order RK method - apparently it's really good
    integrator.set_f_params(m_earth)  # use earth mass
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


## Here I will add a converter that can generate the orbit elements from cartesian elements
## This is intended to take an input state in cartesian coordinates and change it into the state vector


## For now circular orbits are broken, and orbits with zero inclination have ambiguous RAAN and whatnot - I'm just going to convert it to polar maybe
def cartesianToOrbitalElements(
    cartesian_state_vector, central_body_mass, isMeanAnomaly=False
):
    ## Split into a position and a velocity vector
    position = cartesian_state_vector[:3]
    velocity = cartesian_state_vector[3:6]

    ## standard gravitational parameter calculation, please enter mass in kg
    G = 6.67 * 10**-11  # N*m^2/kg^2
    mu = G * central_body_mass

    ## I give credit to René Schwarz, thank you
    ## Find the orbital momentum vector h

    h = np.cross(position, velocity)
    ##Obtain the eccentricity vector e
    e_vector = (np.cross(velocity, h) / mu) - (position / np.linalg.norm(position))
    ## Determine n, the vector pointing towards the ascending node

    n = np.transpose([-h[1], h[0], 0])

    ## determine the true anomaly

    true_anomaly = 0
    if np.dot(position, velocity) >= 0:
        true_anomaly = np.arccos(
            np.dot(e_vector, position)
            / (np.linalg.norm(e_vector) * np.linalg.norm(position))
        )
    else:
        true_anomaly = 2 * np.pi - (
            np.arccos(
                np.dot(e_vector, position)
                / (np.linalg.norm(e_vector) * np.linalg.norm(position))
            )
        )

    # We calculate the orbital inclination by using the orbital momentum vector

    i = np.arccos(h[2] / np.linalg.norm(h))

    # Eccentricity is just the magnitude of the eccentricity vector

    e = np.linalg.norm(e_vector)

    ## We also need the eccentric anomaly

    eccentric_anomaly = 2 * np.arctan(
        np.tan(true_anomaly / 2) / np.sqrt((1 + e) / (1 - e))
    )

    ## Right Ascension of the ascending node

    RA_ascending_node = 0

    if n[1] >= 0:
        RA_ascending_node = np.arccos(n[0] / np.linalg.norm(n))
    else:
        RA_ascending_node = 2 * np.pi - np.arccos(n[0] / np.linalg.norm(n))

    # Argument of periapsis

    arg_peri = 0

    if e_vector[2] >= 0:
        arg_peri = np.arccos(
            np.dot(n, e_vector) / (np.linalg.norm(n) * np.linalg.norm(e_vector))
        )
    else:
        arg_peri = 2 * np.pi - (
            np.arccos(
                np.dot(n, e_vector) / (np.linalg.norm(n) * np.linalg.norm(e_vector))
            )
        )

    ## Compute the mean anomaly, in case you want it

    mean_anomaly = eccentric_anomaly - e * np.sin(eccentric_anomaly)

    ## Compute semimajor axis

    a = 1 / ((2 / np.linalg.norm(position)) - (np.linalg.norm(velocity) ** 2 / mu))
    ## I need a little catch all for when e is zero- this is probably wrong but whatever
    if e == 0:
        RA_ascending_node = (
            np.arccos(n[0] / np.linalg.norm(n)) if np.linalg.norm(n) != 0 else 0
        )
        arg_peri = 0
        # for true anomaly I will manually set a zero point
        # to do this we will just find the angle of wherever it's at using circle math
        # Since orbits are always at least two dimensional if you don't count time and some other things
        # we should only have to consider two cardinal directions
        # This is a bandaid I have not thought too much about but it should be good enough, the orbit is perfectly circular I can
        # do whatever I want
        if position[0] != 0 or velocity[0] != 0:
            true_anomaly = np.cos(position[0] / a)
        else:
            true_anomaly = np.sin(position[2] / a)
    ## we output semimajor axis, eccentricity, inclination, RAAN, Argument of periapsis, true anomaly (or mean anomaly whichever you want)
    if isMeanAnomaly:
        return [a, e, i, RA_ascending_node, arg_peri, mean_anomaly]
    else:
        return [a, e, i, RA_ascending_node, arg_peri, true_anomaly]


## Chat GPT just spit this out so I'll just check if it works - I had it do the other way to check and it seemed pretty close
## Chat was right, I just had to change a couple of things!
## Essentially this works some circle magic assuming its in a nice plane, then rotates it back to reality


def orbitalElementsToCartesian(orbital_state_vector, central_body_mass):
    G = 6.67 * 10**-11  # N*m^2/kg^2
    mu = G * central_body_mass

    a = orbital_state_vector[0]
    e = orbital_state_vector[1]
    i = orbital_state_vector[2]
    Omega = orbital_state_vector[3]  # RAAN
    omega = orbital_state_vector[4]  # perigee
    nu = orbital_state_vector[5]

    # Compute semi-latus rectum
    p = a * (1 - e**2)

    # Calculate position and velocity in the orbital plane
    r = p / (1 + e * np.cos(nu))
    x_p = r * np.cos(nu)
    y_p = r * np.sin(nu)
    v_x_p = -np.sqrt(mu / p) * np.sin(nu)
    v_y_p = np.sqrt(mu / p) * (e + np.cos(nu))

    # Transformation matrices
    ## Apparently all inputs have to be negative to work properly
    R3_Omega = np.array(
        [
            [np.cos(-Omega), np.sin(-Omega), 0],
            [-np.sin(-Omega), np.cos(-Omega), 0],
            [0, 0, 1],
        ]
    )

    R1_i = np.array(
        [[1, 0, 0], [0, np.cos(-i), np.sin(-i)], [0, -np.sin(-i), np.cos(-i)]]
    )

    R3_omega = np.array(
        [
            [np.cos(-omega), np.sin(-omega), 0],
            [-np.sin(-omega), np.cos(-omega), 0],
            [0, 0, 1],
        ]
    )

    # Position and velocity vectors in the orbital frame
    r_orbital = np.array([x_p, y_p, 0])
    v_orbital = np.array([v_x_p, v_y_p, 0])

    # Convert to the inertial frame
    r_inertial = R3_Omega @ R1_i @ R3_omega @ r_orbital
    v_inertial = R3_Omega @ R1_i @ R3_omega @ v_orbital

    return [
        r_inertial[0],
        r_inertial[1],
        r_inertial[2],
        v_inertial[0],
        v_inertial[1],
        v_inertial[2],
    ]
