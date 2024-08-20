from orbitPropagator import twoBodyProp
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
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

training_state_array = twoBodyProp(
    6.371 * 10**6 + altitude_train,  # radius of the earth plus however many meters
    0,
    0,
    0,
    np.sqrt((G * m_earth) / ((6.371 * 10**6) + altitude_train)),
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


# Outputs of arrays are now: x, y, z, vx, vy, vz, time
## NOW we try and make this work, they all have the same shape

train_features = training_state_array[:, 6]
test_features = testing_state_array[:, 6]

## This right here is why I went through all that trouble to change the output of my function in orbitPropagator,
# turns out this is the solution!!!
train_features = train_features.reshape(-1, 1)
test_features = test_features.reshape(-1, 1)

train_labels = training_state_array[:, :6]
test_labels = testing_state_array[:, :6]


def build_and_compile_model(norm):
    model = keras.Sequential(
        [
            norm,
            layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(1),
        ]
    )

    model.compile(loss="mean_absolute_error", optimizer=tf.keras.optimizers.Adam(0.001))
    return model


train_features_normalizer = layers.Normalization(
    axis=None,
)
train_features_normalizer.adapt(train_features)

dnn_train_features_model = build_and_compile_model(train_features_normalizer)
history = dnn_train_features_model.fit(
    train_features, train_labels, validation_split=0.2, verbose=0, epochs=100
)


def plot_loss(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.ylim([0, 10])
    plt.xlabel("Epoch")
    plt.ylabel("Error [X]")
    plt.legend()
    plt.grid(True)


# plot_loss(history)

## Import graphing from last time:
x = np.linspace(0, 12000, 100)
y = dnn_train_features_model.predict(x)

# plotter(y)

test_results = dnn_train_features_model.evaluate(test_features, test_labels)

print(test_results)
