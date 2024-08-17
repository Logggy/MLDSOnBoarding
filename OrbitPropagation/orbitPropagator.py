import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

## Here we will set up the propagator we will use in this task!

## The goal here is simple - I need to set up a propagator that takes the input of x, y, z, vx, vy, vz
## and outputs an orbit

## First we will make a propagator that assumes 2BD


def twoBodyProp(x, y, z, vx, vy, vz):
    ## Lets establish some constants
    G = 6.67 * 10**-11  # N*m^2/kg^2
    m_earth = 5.972 * 10**24  # kg
    r_earth = 6.371 * 10**6  # m
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
    ## 1. set a condition where the propagation stops when the initial x, y, z position is passed (after some time has passed) - failed too sensitive
    ## 2. If you know the eccentricity and the semi-major axis then you can stop when the orbit encompasses the correct area - too complicated probably
    ## 3. rework 1 but how??? Flag the FIRST time slope changes from positive to negative!!! (I thought of this myself!!!!!)
    ## I wonder what the right answer is lol, its probably just finding the orbital parameters then you don't even have to propagate

    ## set up our integrator and associated variables
    integrator = integrate.ode(twoBodyDifEq)
    integrator.set_integrator(
        "dop853"
    )  # use 8th order RK method - apparently it's really good
    integrator.set_f_params(m_earth)  # use earth mass
    integrator.set_initial_value(initial_state_vector, 0)
    dt = 100  # arbitrary
    state_array = np.array([initial_state_vector])
    # time_array = np.array([0])
    i = 1
    while integrator.successful() and i < 100:
        print(i)
        integrator.integrate(integrator.t + dt)
        # time_array = np.append(time_array, [[integrator.t]], axis=0) # dont need
        state_array = np.append(state_array, [integrator.y], axis=0)
        i += 1
        print(i)
        if (
            initial_state_vector[0] == state_array[i - 1, 0]
            and initial_state_vector[1] == state_array[i - 1, 1]
        ):
            break

    return state_array


G = 6.67 * 10**-11  # N*m^2/kg^2
m_earth = 5.972 * 10**24  # kg

print(
    twoBodyProp(
        6.371 * 10**6 + 500,
        0,
        0,
        0,
        np.sqrt((G * m_earth) / ((6.371 * 10**6) + 500)),
        0,
    )
)
