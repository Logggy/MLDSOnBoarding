import numpy as np
import scipy as sp
import scipy.integrate as integrate

# To make it look neater I am just going to make an array with all of the information on each body -
# naturally this will make N the length of the array

# Arrays should be arranged [mass, position, velocity]
table1_info = [
    [1e24, 2e6, 0, 0, 0, 5000, 0],
    [1e24, -2e6, 0, 0, 0, -5000, 0],
    [1e24, 4e6, 0, 0, 0, -5000, 3000],
    [1e24, -4e6, 0, 0, 0, 5000, -3000],
]


def nBodyProp(body_info_array, int_time):
    G = 6.67 * 10**-11  # N * m^2/kg^2
    N = len(body_info_array)

    state_vector = body_info_array[:, 1:7]
    mass = body_info_array[:, 0]

    ## Since we'll want to integrate we need to set up the differential equations
    ## I suppose we will want to set this up in a way where we can integrate
    def nBodyDiffEQ(t, state_vector, mass, N):
        r_norm = np.linalg.norm(state_vector[N, :3] - state_vector[:3])
        vector_sum = 0
        for i in range(N):
            if i != N:
                r_norm = np.linalg.norm(state_vector[N, :3] - state_vector[i, :3])
                vector_sum += (
                    mass[N]
                    * mass[i]
                    * (state_vector[N, :3] - state_vector[i, :3])
                    / r_norm**3
                )

        ax, ay, az = (G / mass[N]) * vector_sum
        return [
            state_vector[N, 3],
            state_vector[N, 4],
            state_vector[N, 5],
            ax,
            ay,
            az,
        ]

    # Now that we can find the acceleration for each body we can move on
    ## set up our integrator and associated variables
    integrator = integrate.ode(nBodyDiffEQ)
    integrator.set_integrator(
        "dop853"
    )  # use 8th order RK method - apparently it's really good
    dt = 10
    i = 1
    integrator.set_initial_value(state_vector, 0)
    state_array = np.array()
    time_array = np.array()
    # What we're doing here is creating a full array of each propagated orbit - the first value being at the first
    # time step after zero for each
    for i in range(N):
        while integrator.successful():
            integrator.set_f_params(mass, i)  # use earth mass
            integrator.integrate(integrator.t + dt)
            time_array = np.append(time_array, [integrator.t], axis=0)
            state_array = np.append(state_array, [integrator.y], axis=0)

            if i * dt > int_time:
                break

    return state_array, time_array
