import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import rebound

# To make it look neater I am just going to make an array with all of the information on each body -
# naturally this will make N the length of the array
# I am going to use rebounds symplectic integrator
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
    acceleration_array = np.array([[0, 0, 0]])  # To store accelerations
    # What we're doing here is creating a full array of each propagated orbit - the first value being at the first
    # time step after zero for each
    for i in range(N):
        state_array = np.append(state_array, [state_vector[i]], axis=0)
        time_array = np.append(time_array, [0], axis=0)
        acceleration_array = np.append(
            acceleration_array, [[0, 0, 0]], axis=0
        )  # Initial acceleration
        integrator.set_initial_value(state_vector[i], 0)
        integrator.set_f_params(state_vector, mass, i, N)
        j = 1
        while integrator.successful():
            if j * dt > int_time:
                break
            integrator.integrate(integrator.t + dt)
            time_array = np.append(time_array, [integrator.t], axis=0)
            state_array = np.append(state_array, [integrator.y], axis=0)
            current_acceleration = integrator.f(
                integrator.t, integrator.y, state_vector, mass, i, N
            )[3:]
            acceleration_array = np.append(
                acceleration_array, [current_acceleration], axis=0
            )
            j += 1
    # Must remove the first array element
    state_array = state_array[1:, :]
    time_array = time_array[1:]
    acceleration_array = acceleration_array[1:]
    # note that the number of outputs in an array is N(int_time/dt) + N
    return state_array, time_array, acceleration_array


state_array, time_array, acceleration_array = nBodyProp(table1_info, 20000)
N = 4
dt = 10
int_time = 20000
body1 = state_array[: int(1 * (int_time / dt) + 1)]
body2 = state_array[int(1 * (int_time / dt) + 1) : int(2 * (int_time / dt) + 2)]
body3 = state_array[int(2 * (int_time / dt) + 2) : int(3 * (int_time / dt) + 3)]
body4 = state_array[int(3 * (int_time / dt) + 3) : int(4 * (int_time / dt) + 4)]
bodies = [body1, body2, body3, body4]
bodies_masses = [1e24, 1e24, 1e24, 1e24]
body1_accel = acceleration_array[: int(1 * (int_time / dt) + 1)]
body2_accel = acceleration_array[
    int(1 * (int_time / dt) + 1) : int(2 * (int_time / dt) + 2)
]
body3_accel = acceleration_array[
    int(2 * (int_time / dt) + 2) : int(3 * (int_time / dt) + 3)
]
body4_accel = acceleration_array[
    int(3 * (int_time / dt) + 3) : int(4 * (int_time / dt) + 4)
]
bodies_accel = [body1_accel, body2_accel, body3_accel, body4_accel]
plt.style.use("dark_background")
fig = plt.figure(figsize=(10, 10))
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))

ax = fig.add_subplot(111, projection="3d")
for i in range(N):
    # Trajectory
    ax.plot(
        bodies[i][:, 0],
        bodies[i][:, 1],
        bodies[i][:, 2],
        "w",
        label="Body Trajectory",
    )
    ax.plot(
        bodies[i][0, 0],
        bodies[i][0, 1],
        bodies[i][0, 2],
        "ro",
        label="Body Initial Condition",
    )

plt.show()

## I'll try to plot the energy here
G = 6.67 * 10**-11  # N * m^2/kg^2
potential_e = np.zeros(int(1 * (int_time / dt) + 1))
kinetic_e = np.zeros(int(1 * (int_time / dt) + 1))

for k in range(int(1 * (int_time / dt) + 1)):
    potential_e_sum = 0
    kinetic_e_sum = 0
    for i in range(N):
        kinetic_e_sum += (
            0.5
            * bodies_masses[i]
            * np.linalg.norm(bodies[i][k, 3:])
            * np.linalg.norm(bodies_accel[i][k])
        )
        for j in range(N):
            if i != j:
                potential_e_sum += (
                    0.5
                    * G
                    * bodies_masses[i]
                    * bodies_masses[j]
                    / np.linalg.norm(bodies[i][k, :3] - bodies[j][k, :3])
                )
    potential_e[k] = potential_e_sum
    kinetic_e[k] = kinetic_e_sum

total_e = kinetic_e - potential_e

plt.plot(total_e)

plt.show()
