import numpy as np
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
    body_array = np.array(body_info_array)
    state_vector = body_array[:, 1:7]
    mass = body_array[:, 0]

    ## Since we'll want to integrate we need to set up the differential equations
    ## I suppose we will want to set this up in a way where we can integrate
    def nBodyDiffEQ(t, state_vector_primary, state_vector_n, mass, Nprim, Nbod):
        vector_sum = np.array([0.0, 0.0, 0.0])
        for i in range(Nbod):
            if i != Nprim:
                r_norm = np.linalg.norm(
                    state_vector_primary[:3] - state_vector_n[i, :3]
                )
                vector_sum += (
                    mass[Nprim]
                    * mass[i]
                    * (state_vector_primary[:3] - state_vector_n[i, :3])
                    / r_norm**3
                )

        ax, ay, az = (G / mass[Nprim]) * vector_sum
        return [
            state_vector_primary[3],
            state_vector_primary[4],
            state_vector_primary[5],
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
    state_array = np.array([[0, 0, 0, 0, 0, 0]])
    time_array = np.array([0])
    # What we're doing here is creating a full array of each propagated orbit - the first value being at the first
    # time step after zero for each
    for i in range(N):
        integrator.set_initial_value(state_vector[i], 0)
        integrator.set_f_params(state_vector, mass, i, N)
        j = 1
        while integrator.successful():
            if j * dt > int_time:
                break
            integrator.integrate(integrator.t + dt)
            time_array = np.append(time_array, [integrator.t], axis=0)
            state_array = np.append(state_array, [integrator.y], axis=0)
            j += 1

    return state_array, time_array


state_array, time_array = nBodyProp(table1_info, 20000)
