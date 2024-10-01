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
    ## I now need to compute the accelerations for each object at each time step
    def nBodyDiffEQ(r_vectors, masses, N):
        r_vectors = state_vector[:, :3]
        accelerations = np.zeros_like(r_vectors)
        for i in range(N):
            for j in range(N):
                if i != j:
                    r_ij = r_vectors[j] - r_vectors[i]
                    r_norm = np.linalg.norm(r_ij)
                    # Gravitational acceleration contribution
                    accelerations[i] += G * masses[j] * r_ij / r_norm**3
        return accelerations

    def leapfrog_integration(state_vector, masses, dt, int_time):
        N = len(masses)  # Number of bodies
        num_steps = int(int_time / dt)
        r_vector = state_vector[:, :3]
        v_vector = state_vector[:, 3:]
        # Initialize arrays to store positions and velocities at each step
        position_history = np.zeros((num_steps + 1, N, 3))
        velocity_history = np.zeros((num_steps + 1, N, 3))
        acceleration_history = np.zeros((num_steps + 1, N, 3))
        time_array = np.zeros(num_steps + 1)

        # Store initial conditions
        position_history[0] = r_vector
        velocity_history[0] = v_vector

        # Leapfrog integration loop
        ## Since I knew nothing about symplectic integrators chatgpt educated me on the very simple leap-frog algorithm
        for step in range(1, num_steps + 1):

            # Update accelerations based on the new positions
            new_accelerations = nBodyDiffEQ(r_vector, masses, N)
            # Drift step: update positions by half a time step

            # Kick step: update velocities by a full time step using the new accelerations
            v_vector += new_accelerations * (dt / 2)

            r_vector += v_vector * (dt)

            # Update accelerations based on the new positions
            new_accelerations = nBodyDiffEQ(r_vector, masses, N)

            # Kick step: update velocities by a full time step using the new accelerations
            v_vector += new_accelerations * (dt / 2)

            # Store the updated positions and velocities
            position_history[step] = r_vector
            velocity_history[step] = v_vector
            acceleration_history[step] = new_accelerations
            time_array[step] = time_array[step - 1] + dt

        return position_history, velocity_history, acceleration_history, time_array

    # Run the Leapfrog integrator
    positions, velocities, accelerations, time_array = leapfrog_integration(
        state_vector, mass, dt=1, int_time=int_time
    )
    return positions, velocities, accelerations, time_array


positions, velocities, accelerations, time_array = nBodyProp(table1_info, 20000)
N = 4
dt = 1
int_time = 20000
body1_position = positions[:, 0]
body2_position = positions[:, 1]
body3_position = positions[:, 2]
body4_position = positions[:, 3]
body1_velocity = velocities[:, 0]
body2_velocity = velocities[:, 1]
body3_velocity = velocities[:, 2]
body4_velocity = velocities[:, 3]
bodies = [body1_position, body2_position, body3_position, body4_position]
bodies_velocity = [body1_velocity, body2_velocity, body3_velocity, body4_velocity]
bodies_masses = [1e24, 1e24, 1e24, 1e24]
body1_accel = accelerations[:, 0]
body2_accel = accelerations[:, 1]
body3_accel = accelerations[:, 2]
body4_accel = accelerations[:, 3]
bodies_accel = [body1_accel, body2_accel, body3_accel, body4_accel]
plt.style.use("dark_background")
fig = plt.figure(figsize=(10, 10))
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))

ax = fig.add_subplot(111, projection="3d")
colors = ["r", "cyan", "orange", "blue"]
for i in range(N):
    # Trajectory
    ax.plot(
        bodies[i][:, 0],
        bodies[i][:, 1],
        bodies[i][:, 2],
        colors[i],
        label="Body Trajectory",
    )
    ax.plot(
        bodies[i][0, 0],
        bodies[i][0, 1],
        bodies[i][0, 2],
        color="lime",
        marker="o",
        label="Body Initial Condition",
    )
ax.set_xlabel(["X (m)"])
ax.set_ylabel(["Y (m)"])
ax.set_zlabel(["Z (m)"])
plt.title("Four Body Problem Propagation")
plt.show()

fig, axes = plt.subplots(3, 1, figsize=(10, 12))

# Define colors for each body
colors = ["red", "cyan", "orange", "blue"]
labels = ["Body 1", "Body 2", "Body 3", "Body 4"]

# Plot x-components
axes[0].plot(body1_position[:, 0], color=colors[0], label=labels[0])
axes[0].plot(body2_position[:, 0], color=colors[1], label=labels[1])
axes[0].plot(body3_position[:, 0], color=colors[2], label=labels[2])
axes[0].plot(body4_position[:, 0], color=colors[3], label=labels[3])
axes[0].set_ylabel("X Position (m)")
axes[0].legend()
axes[0].grid(True)

# Plot y-components
axes[1].plot(body1_position[:, 1], color=colors[0], label=labels[0])
axes[1].plot(body2_position[:, 1], color=colors[1], label=labels[1])
axes[1].plot(body3_position[:, 1], color=colors[2], label=labels[2])
axes[1].plot(body4_position[:, 1], color=colors[3], label=labels[3])
axes[1].set_ylabel("Y Position (m)")
axes[1].legend()
axes[1].grid(True)

# Plot z-components
axes[2].plot(body1_position[:, 2], color=colors[0], label=labels[0])
axes[2].plot(body2_position[:, 2], color=colors[1], label=labels[1])
axes[2].plot(body3_position[:, 2], color=colors[2], label=labels[2])
axes[2].plot(body4_position[:, 2], color=colors[3], label=labels[3])
axes[2].set_ylabel("Z Position (m)")
axes[2].legend()
axes[2].grid(True)

# Set common labels and title
fig.suptitle("Body Positions Over Time")
fig.text(0.5, 0.04, "Time Steps (seconds)", ha="center")
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
            0.5 * bodies_masses[i] * np.linalg.norm(bodies_velocity[i][k]) ** 2
        )
        for j in range(N):
            if i != j:
                potential_e_sum += (
                    0.5
                    * G
                    * bodies_masses[i]
                    * bodies_masses[j]
                    / np.linalg.norm(bodies[j][k] - bodies[i][k])
                )
    potential_e[k] = potential_e_sum
    kinetic_e[k] = kinetic_e_sum

total_e = kinetic_e - potential_e

plt.plot(total_e)
plt.title("Total System Energy Over Time")
plt.xlabel("Time Steps (seconds)")
plt.ylabel("Energy (kg * (m/s)^2)")
plt.show()

## Now I just need
angular_momentum = np.zeros(int(1 * (int_time / dt) + 1))

for i in range(int(1 * (int_time / dt) + 1)):
    sum = 0
    for j in range(N):
        sum += bodies_masses[j] * np.cross(bodies[j][i], bodies_velocity[j][i])
    angular_momentum[i] = np.linalg.norm(sum)

plt.plot(angular_momentum)
plt.title("Total System Angular Momentum Over Time")
plt.xlabel("Time Steps (seconds)")
plt.ylabel("Angular Momentum (kg * m^2/s)")
plt.show()
