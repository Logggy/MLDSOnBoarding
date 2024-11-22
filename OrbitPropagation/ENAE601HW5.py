import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

## We shall now create a CR3BP propagator

mu = 0.012150585609624
du = 384747.96285603


def CR3BPProp(
    cartesian_state_vector,
    mu,
    du,
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
    x1 = -mu
    x2 = 1 - mu
    ## Now we propagate
    ## In this solution we know that F = GMm/r^2
    ## Fun fact - I did not know that the norm of the vector is its magnitude (I thought it meant normalization)
    ## Essentially we already know the solutions to Newtons Equations namely - a = -GMr / mag_r^3, where a and r are vectors
    ## So now it's easy - we have the solutions of the original function (x, y, z, vx, vy, vz)
    ## AND their derivatives (vx, vy, vz, ax, ay, az), all we need to do is integrate to solve the ODE

    def CR3BPDifEq(t, state_vector, M):
        r13 = np.linalg.norm([x1 - state_vector[0], state_vector[1], state_vector[2]])
        r23 = np.linalg.norm([x2 - state_vector[0], state_vector[1], state_vector[2]])

        ax = (
            (2 * state_vector[4])
            + state_vector[0]
            - ((1 - mu) * ((state_vector[0] - x1) / (r13**3)))
            - ((mu * (state_vector[0] - x2)) / (r23**3))
        )
        ay = (
            -(2 * state_vector[3])
            + state_vector[1]
            - (((1 - mu) / (r13**3)) + (mu / (r23**3))) * state_vector[1]
        )
        az = -(((1 - mu) / (r13**3)) + (mu / (r23**3))) * state_vector[2]
        return [state_vector[3], state_vector[4], state_vector[5], ax, ay, az]

    ## set up our integrator and associated variables
    integrator = integrate.ode(CR3BPDifEq)
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


orbitA = CR3BPProp(
    [1.2, 0, 0, 0, -1.049357509830343, 0],
    0.012150585609624,
    0,
    oneOrbit=False,
    timedOrbit=6.192169331319632,
    time_step=0.001,
    export_time=True,
)

print(orbitA[-1])


def plotter(
    state_array,
    mu,
    plot2d=False,
    primarysecondary=False,
    primary=[],
    secondary=[],
    title="",
    dimensional=1,
):
    radius_earth = 6378000  # m
    G = 6.67 * 10**-11  # N*m^2/kg^2

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
        "b",
        label="Craft Trajectory",
    )
    ax.plot(
        state_array[0, 0],
        state_array[0, 1],
        state_array[0, 2],
        "ro",
        label="Initial Condition",
    )

    ## Now I'll plot the vector of the initial condition
    # ax.quiver(
    #     state_array[0, 0],
    #     state_array[0, 1],
    #     state_array[0, 2],
    #     state_array[0, 3],
    #     state_array[0, 4],
    #     state_array[0, 5],
    #     color="r",
    #     label="Initial Condition Vector (m / s)",
    # )

    ## Set limits + Graph Specs
    x1 = -mu
    x2 = 1 - mu
    if np.abs(x2 * dimensional) > np.abs(np.max(state_array[:, :3])):
        graph_limit = x2 * dimensional
    else:
        graph_limit = np.max(np.abs(state_array[:, :3]))

    ax.set_xlim([-graph_limit, graph_limit])
    ax.set_ylim([-graph_limit, graph_limit])
    ax.set_zlim([-graph_limit, graph_limit])

    ax.set_xlabel(["X (nondimensional)"])
    ax.set_ylabel(["Y (nondimensional)"])
    ax.set_zlabel(["Z (nondimensional)"])
    ax.scatter(x1 * dimensional, 0, label="Primary Mass")
    ax.scatter(x2 * dimensional, 0, label="Secondary Mass")
    ax.set_title(title + " Rotating Frame")
    if primarysecondary:
        ax.plot(primary[:, 0], primary[:, 1], primary[:, 2], label="Primary Mass")
        ax.plot(
            secondary[:, 0], secondary[:, 1], secondary[:, 2], label="Secondary Mass"
        )
        if np.max(np.abs(secondary[:, :])) > np.max(state_array[:, :]):
            graph_limit = np.max(np.abs(secondary[:, :]))
        else:
            graph_limit = np.max(np.abs(state_array[:, :]))

        ax.set_xlim([-graph_limit, graph_limit])
        ax.set_ylim([-graph_limit, graph_limit])
        ax.set_zlim([-graph_limit, graph_limit])
        ax.set_xlabel(["X (km)"])
        ax.set_ylabel(["Y (km)"])
        ax.set_zlabel(["Z (km)"])

        ax.set_title(title + " Inertial Frame")
    plt.legend()
    plt.show()

    if plot2d:
        plt.plot(state_array[:, 0], state_array[:, 1], label="Orbit Trajectory")
        plt.scatter(x1, 0, label="Primary Mass")
        plt.scatter(x2, 0, label="Secondary Mass")
        plt.grid()
        plt.legend()
        plt.xlabel("X (Nondimensional)")
        plt.ylabel("Y (Nondimensional)")
        plt.title("CR3BP Orbit A, Rotating Frame")
        plt.show()


plotter(orbitA, mu, plot2d=True, title="CR3BP Orbit A")

## Question 3
# We can already propagate in the canonical frame, so we just take what we have and rotate it!
orbitBinitial = [-0.08, -0.03, 0.01, 3.5, -3.1, -0.1]
orbitCinitial = [0.05, -0.05, 0, 4.0, 2.6, 0]
orbitDinitial = [0.83, 0, 0.114062816271683, 0, 0.229389507175582, 0]
orbitEinitial = [-0.05, -0.02, 0, 4.09, -5.27, 0]
x1 = -mu
x2 = 1 - mu
orbitB = CR3BPProp(
    orbitBinitial,
    0.012150585609624,
    0,
    oneOrbit=False,
    timedOrbit=26,
    time_step=0.01,
    export_time=True,
)

orbitC = CR3BPProp(
    orbitCinitial,
    0.012150585609624,
    0,
    oneOrbit=False,
    timedOrbit=25,
    time_step=0.01,
    export_time=True,
)

orbitD = CR3BPProp(
    orbitDinitial,
    0.012150585609624,
    0,
    oneOrbit=False,
    timedOrbit=15,
    time_step=0.01,
    export_time=True,
)

orbitE = CR3BPProp(
    orbitEinitial,
    0.012150585609624,
    0,
    oneOrbit=False,
    timedOrbit=15,
    time_step=0.01,
    export_time=True,
)


## Rotating orbits
plotter(orbitB, mu, title="CR3BP Orbit B")
plotter(orbitC, mu, title="CR3BP Orbit C")
plotter(orbitD, mu, title="CR3BP Orbit D")
plotter(orbitE, mu, title="CR3BP Orbit E")


def CR3BPtoInertial(orbit, DU, mu):
    # You just take the orbit and rotate it
    def rotation(t):
        return np.matrix(
            [[np.cos(t), np.sin(t), 0], [-np.sin(t), np.cos(t), 0], [0, 0, 1]]
        )

    orbitinertial = np.zeros((len(orbit), 3))
    primaryinertial = np.zeros((len(orbit), 3))
    secondaryinertial = np.zeros((len(orbit), 3))
    x1 = [-mu, 0, 0]
    x2 = [1 - mu, 0, 0]
    for i in range(len(orbit)):
        orbitinertial[i] = rotation(orbit[i, 6]) @ np.transpose(orbit[i, :3])
        primaryinertial[i] = rotation(orbit[i, 6]) @ np.transpose(x1)
        secondaryinertial[i] = rotation(orbit[i, 6]) @ np.transpose(x2)

    return orbitinertial * du, primaryinertial * du, secondaryinertial * du


orbitBinertial, orbitBprimaryinertial, orbitBsecondaryinertial = CR3BPtoInertial(
    orbitB, du, mu
)

plotter(
    orbitBinertial,
    mu,
    primarysecondary=True,
    primary=orbitBprimaryinertial,
    secondary=orbitBsecondaryinertial,
    title="CR3BP Orbit B",
    dimensional=du,
)

orbitCinertial, orbitCprimaryinertial, orbitCsecondaryinertial = CR3BPtoInertial(
    orbitC, du, mu
)

plotter(
    orbitCinertial,
    mu,
    primarysecondary=True,
    primary=orbitCprimaryinertial,
    secondary=orbitCsecondaryinertial,
    title="CR3BP Orbit C",
    dimensional=du,
)

orbitDinertial, orbitDprimaryinertial, orbitDsecondaryinertial = CR3BPtoInertial(
    orbitD, du, mu
)

plotter(
    orbitDinertial,
    mu,
    primarysecondary=True,
    primary=orbitDprimaryinertial,
    secondary=orbitDsecondaryinertial,
    title="CR3BP Orbit D",
    dimensional=du,
)

orbitEinertial, orbitEprimaryinertial, orbitEsecondaryinertial = CR3BPtoInertial(
    orbitE, du, mu
)

plotter(
    orbitEinertial,
    mu,
    primarysecondary=True,
    primary=orbitEprimaryinertial,
    secondary=orbitEsecondaryinertial,
    title="CR3BP Orbit E",
    dimensional=du,
)

orbitAinertial, orbitAprimaryinertial, orbitAsecondaryinertial = CR3BPtoInertial(
    orbitA, du, mu
)

plotter(
    orbitAinertial,
    mu,
    primarysecondary=True,
    primary=orbitAprimaryinertial,
    secondary=orbitAsecondaryinertial,
    title="CR3BP Orbit A",
    dimensional=du,
)
## Question 4


def c_generator(state_initial, mu):
    x1 = -mu
    x2 = 1 - mu
    r13 = np.linalg.norm([x1 - state_initial[0], state_initial[1], state_initial[2]])
    r23 = np.linalg.norm([x2 - state_initial[0], state_initial[1], state_initial[2]])
    c = (
        -(state_initial[3] ** 2 + state_initial[4] ** 2 + state_initial[5] ** 2)
        + (state_initial[0] ** 2 + state_initial[1] ** 2 + state_initial[2] ** 2)
        + ((2 * (1 - mu)) / r13)
        + ((2 * mu) / r23)
    )
    return c


cb = c_generator(orbitBinitial, mu)
cd = c_generator(orbitDinitial, mu)

## we need to find the values for x, y, r1, and r2 that satisfy
# (state_initial[0] ** 2 + state_initial[1] ** 2 + state_initial[2] ** 2) + ((2 * (1 - mu)) / r13) + (2 * mu) / r23


def zeroCurve(c, mu):
    x = np.linspace(-np.sqrt(c), np.sqrt(c), 2000)
    y = np.linspace(-np.sqrt(c), np.sqrt(c), 2000)
    x1 = -mu
    x2 = 1 - mu
    zerox = []
    zeroy = []
    for i in range(len(x)):
        for j in range(len(y)):
            r13 = np.linalg.norm([x1 - x[i], y[j]])
            r23 = np.linalg.norm([x2 - x[i], y[j]])

            if (x[i] ** 2 + y[j] ** 2) + ((2 * (1 - mu)) / r13) + (
                (2 * mu) / r23
            ) - c >= 0:
                zerox.append(x[i])
                zeroy.append(y[j])
    return zerox, zeroy


zeroxb, zeroyb = zeroCurve(cb, mu)
zeroxd, zeroyd = zeroCurve(cd, mu)
plt.scatter(zeroxb, zeroyb, s=1, label="V**2 >= 0")
plt.plot(orbitB[:, 0], orbitB[:, 1], color="r", label="Orbit B Trajectory")
plt.xlabel("X (nondimensional)")
plt.ylabel("Y (nondimensional)")
plt.scatter(x1, 0, s=5, label="Primary Mass")
plt.scatter(x2, 0, s=5, color="k", label="Secondary Mass")
plt.title("Zero Velocity Curve + XY Trajectory Orbit B")
plt.legend()
plt.show()

plt.scatter(zeroxd, zeroyd, s=1, label="V**2 >= 0")
plt.plot(orbitD[:, 0], orbitD[:, 1], color="r", label="Orbit D Trajectory")
plt.scatter(x1, 0, s=5, label="Primary Mass")
plt.scatter(x2, 0, s=5, color="k", label="Secondary Mass")
plt.xlabel("X (nondimensional)")
plt.ylabel("Y (nondimensional)")
plt.title("Zero Velocity Curve + XY Trajectory Orbit D")
plt.legend()
plt.show()
