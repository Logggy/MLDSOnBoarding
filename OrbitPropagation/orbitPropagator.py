import numpy as np
import scipy as sp

## Here we will set up the propagator we will use in this task!

## The goal here is simple - I need to set up a propagator that takes the input of x, y, z, vx, vy, vz
## and outputs an orbit

## First we will make a propagator that assumes 2BD


def twoBodyProp(x, y, z, vx, vy, vz):
    ## Lets establish some constants
    G = 6.67 * 10**-11  # N*m^2/kg^2
    m_earth = 5.972 * 10**24  # kg
    r_earth = 6.371 * 10**6  # m
    state_vector = [x, y, z, vx, vy, vz]

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
