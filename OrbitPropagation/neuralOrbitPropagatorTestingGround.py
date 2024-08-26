from orbitPropagator import (
    twoBodyProp,
    cartesianToOrbitalElements,
    orbitalElementsToCartesian,
)
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from orbitPlotter import plotter


## So this will be using tensorflow and I think I'll just make this into one full script that does everything
## the first thing to do will be to make training datasets - luckily I formatted my two body orbit propagator to do this with ease


# Use the imported orbit propagator to make training and test sets
# Now here's a question - I assume you want to train on a fully complete orbit right?
# Could you do half an orbit and see if it gets the next half?

# Here I'll make a training set that is an orbit 5000km above the earth, and we'll make the test set 8000 km
# We'll make it a little dense too with a dt of 10

G = 6.67 * 10**-11  # N*m^2/kg^2
m_earth = 5.972 * 10**24  # kg
altitude_test = 8 * 10**6  # m
altitude_train = 5 * 10**6  # m

broken_orbit = [
    1.83789060e07,
    9.08747775e-02,
    3.88722922e-01,
    4.04577817e00,
    2.10304796e00,
    1.00097003e00,
]

broken_cartesian_orbit = orbitalElementsToCartesian(broken_orbit, m_earth)
broken_orbit_prop = twoBodyProp(
    broken_cartesian_orbit[0],
    broken_cartesian_orbit[1],
    broken_cartesian_orbit[2],
    broken_cartesian_orbit[3],
    broken_cartesian_orbit[4],
    broken_cartesian_orbit[5],
    export_time=True,
    time_step=50,
)

plotter(broken_orbit_prop)

broken_orbit_mod = [
    1.83789060e07,
    9.08747775e-02,
    np.pi / 4,
    4.04577817e00,
    2.10304796e00,
    1.00097003e00,
]

broken_cartesian_orbit = orbitalElementsToCartesian(broken_orbit_mod, m_earth)
broken_orbit_prop = twoBodyProp(
    broken_cartesian_orbit[0],
    broken_cartesian_orbit[1],
    broken_cartesian_orbit[2],
    broken_cartesian_orbit[3],
    broken_cartesian_orbit[4],
    broken_cartesian_orbit[5],
    export_time=True,
    time_step=50,
)

plotter(broken_orbit_prop)
plt.show()

training_state_array = twoBodyProp(
    6.371 * 10**6 + 5 * 10**6,  # radius of the earth plus however many meters
    0,
    0,
    0,
    np.sqrt((G * 5.972 * 10**24) / ((6.371 * 10**6) + 5 * 10**6)) + 500,
    200,
    time_step=10,
    export_time=True,
)

training_state_array2 = twoBodyProp(
    6.371 * 10**6
    + altitude_train
    + 2000000,  # radius of the earth plus however many meters
    45000,
    4000,
    300,
    np.sqrt(
        (G * m_earth) / ((6.371 * 10**6) + altitude_test)
    ),  # you simply get the two body velocity from this by setting gravitational force equal to centripetal force
    200,
    export_time=True,
)

training_state_array3 = twoBodyProp(
    6.371 * 10**6
    + altitude_train
    + 4000,  # radius of the earth plus however many meters
    0,
    0,
    0,
    np.sqrt(
        (G * m_earth) / ((6.371 * 10**6) + altitude_train + 4000)
    ),  # you simply get the two body velocity from this by setting gravitational force equal to centripetal force
    0,
    export_time=True,
)

testing_state_array = twoBodyProp(
    6.371 * 10**6 + altitude_test,  # radius of the earth plus however many meters
    0,
    0,
    0,
    np.sqrt(
        (G * m_earth) / ((6.371 * 10**6) + altitude_test)
    ),  # you simply get the two body velocity from this by setting gravitational force equal to centripetal force
    0,
    export_time=True,
)

#### Below is the first attempt using only cartesian coordinates, I will now make training and test sets based on orbital elements
# Outputs of arrays are now: x, y, z, vx, vy, vz, time
## NOW we try and make this work, they all have the same shape
## Now we need to make it so the initial condition is known at every step!
# train_features = training_state_array[1:, 6].reshape(
#     -1, 1
# )  # Elapsed time for each step
# train_initial_conditions = np.repeat(
#     training_state_array[0, :6].reshape(1, -1), train_features.shape[0], axis=0
# )

# # squash train_features and train initial conditions into one
# train_features = np.hstack([train_initial_conditions, train_features])

# # train_features = train_features[1:3]
# train_labels = training_state_array[
#     1:, :6
# ]  # Predicting states after the initial condition
# test_features = testing_state_array[1:, 6].reshape(-1, 1)  # Elapsed time for each step
# test_initial_conditions = np.repeat(
#     testing_state_array[0, :6].reshape(1, -1), test_features.shape[0], axis=0
# )
# test_labels = testing_state_array[
#     1:, :6
# ]  # Predicting states after the initial condition

# # train_labels = train_labels[1:3]

# # squash train_features and train initial conditions into one
# test_features = np.hstack([test_initial_conditions, test_features])
## So now we have two inputs - the time, and an array of the initial conditions so the model remembers


## Converge into a huge training set
# training_state_array = np.vstack(
#     [training_state_array1, training_state_array2, training_state_array3]
# )

## Convert the arrays to orbital elements
training_state_array_orbitalE = np.zeros((len(training_state_array), 6))
testing_state_array_orbitalE = np.zeros((len(testing_state_array), 6))
for i in range(len(training_state_array)):
    array = cartesianToOrbitalElements(training_state_array[i], m_earth)
    for j in range(6):
        training_state_array_orbitalE[i, j] = array[j]

for i in range(len(testing_state_array)):
    array = cartesianToOrbitalElements(testing_state_array[i], m_earth)
    for j in range(6):
        testing_state_array_orbitalE[i, j] = array[j]

## the features are the first 5 orbital elements, and the time, the label should probably just be true anomaly

train_features = np.copy(training_state_array_orbitalE)
test_features = np.copy(testing_state_array_orbitalE)

for i in range(len(train_features)):
    train_features[i, -1] = training_state_array[i, -1]

for i in range(len(test_features)):
    test_features[i, -1] = testing_state_array[i, -1]

train_labels = training_state_array_orbitalE[:, -1]
test_labels = testing_state_array_orbitalE[:, -1]
# Normalize train_features (time) and train_initial_conditions (initial state)
train_features_normalizer = layers.Normalization(axis=-1)
train_features_normalizer.adapt(train_features)


# Build the model using keras.Sequential
def build_and_compile_model(norm):
    model = keras.Sequential(
        [
            # Time normalization layer
            norm,
            layers.Dense(30, activation="relu"),
            layers.Dense(30, activation="relu"),
            layers.Dense(1),
        ]
    )

    model.compile(
        loss="mean_absolute_error",
        optimizer=tf.keras.optimizers.Adam(0.1),
        metrics=["accuracy"],
    )
    return model


dnn_train_features_model = build_and_compile_model(train_features_normalizer)
history = dnn_train_features_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=1,
    epochs=6 * 10**2,
)


# Plotting the loss
def plot_loss(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.ylim([0, 10])
    plt.xlabel("Epoch")
    plt.ylabel("Error [X]")
    plt.legend()
    plt.grid(True)


plot_loss(history)
plt.show()

test_results = dnn_train_features_model.evaluate(test_features, test_labels, verbose=0)

test_predictions = dnn_train_features_model.predict(test_features)
test_prediction_array_orbitalE = np.zeros((len(test_predictions), 6))
# print(test_predictions)
for i in range(len(test_predictions)):
    for j in range(6):
        if j < 5:
            test_prediction_array_orbitalE[i, j] = test_features[i, j]
        else:
            test_prediction_array_orbitalE[i, j] = test_predictions[i]
print("Orbital E: ", test_prediction_array_orbitalE)
## Convert to cartesian
test_prediction_array_cartesian = np.zeros((len(test_predictions), 6))
for i in range(len(test_prediction_array_cartesian)):
    element_to_cart = orbitalElementsToCartesian(
        test_prediction_array_orbitalE[i], m_earth
    )
    for j in range(6):
        test_prediction_array_cartesian[i, j] = element_to_cart[j]
print("Cartesian: ", test_prediction_array_cartesian)
plotter(test_prediction_array_cartesian)
plotter(testing_state_array)
plt.show()