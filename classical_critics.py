import tensorflow as tf
from tensorflow.keras.layers import *

class DenseCritic(tf.keras.Model):

    def __init__(self, obs_size, hidden_sizes, activation=None):
        super().__init__()

        self.input_shape_ = (obs_size, )

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.Input(shape=(obs_size,)))

        self.model.add(
            tf.keras.layers.Dense(hidden_sizes[0], input_shape=(obs_size,),
            kernel_initializer='he_uniform',
            activation=activation)
        )

        for k in range(1, len(hidden_sizes)-1):
            self.model.add(
                tf.keras.layers.Dense(hidden_sizes[k],
                input_shape=(hidden_sizes[k-1],),
                kernel_initializer='he_uniform',
                activation=activation)
            )

        # No activation on the last layer !
        self.model.add(
            tf.keras.layers.Dense(1, input_shape=(hidden_sizes[-1],),
            kernel_initializer='he_uniform',
            activation=None)
        )

    def call(self, inputs):
        return self.model(inputs)


class ConvImgCritic(tf.keras.Model):

    def __init__(self, obs_size_x, obs_size_y, obs_size_z,
                 hidden_sizes, activation=None):
        super().__init__()

        self.input_shape_ = (obs_size_x, obs_size_y, obs_size_z)

        self.model = tf.keras.Sequential()
        # self.model.add(tf.keras.Input(shape=shape))

        self.model.add(tf.keras.layers.Conv2D(
            32, 8, strides=4, activation=activation, padding="same"
        ))

        self.model.add(tf.keras.layers.Conv2D(
            64, 4, strides=2, activation=activation, padding="same"
        ))

        self.model.add(tf.keras.layers.Conv2D(
            64, 3, strides=2, activation=activation, padding="same"
        ))

        self.model.add(tf.keras.layers.Flatten())
        self.model.add(
            tf.keras.layers.Dense(hidden_sizes[0],
            kernel_initializer='he_uniform',
            activation=activation)
        )


        for k in range(1, len(hidden_sizes)-1):
            self.model.add(
                tf.keras.layers.Dense(hidden_sizes[k],
                input_shape=(hidden_sizes[k-1],),
                kernel_initializer='he_uniform',
                activation=activation)
            )

        # No activation on the last layer !
        self.model.add(
            tf.keras.layers.Dense(1, input_shape=(hidden_sizes[-1],),
            kernel_initializer='he_uniform',
            activation=None)
        )

    def call(self, inputs):
        return self.model(inputs)


class ConvImgCritic2(tf.keras.Model):

    def __init__(self, obs_size_x, obs_size_y, obs_size_z,
                 hidden_sizes, activation=None):
        super().__init__()

        self.input_shape_ = (obs_size_x, obs_size_y, obs_size_z)

        self.model = tf.keras.Sequential()
        # self.model.add(tf.keras.Input(shape=shape))

        self.model.add(tf.keras.layers.Conv2D(
            32, (3, 3), activation=activation, padding="same"
        ))
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))

        self.model.add(tf.keras.layers.Conv2D(
            64, (3, 3), activation=activation, padding="same"
        ))
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))

        self.model.add(tf.keras.layers.Conv2D(
            128, (3, 3), activation=activation, padding="same"
        ))
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))

        self.model.add(tf.keras.layers.Conv2D(
            256, (3, 3), activation=activation, padding="same"
        ))
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))

        self.model.add(tf.keras.layers.Flatten())
        self.model.add(
            tf.keras.layers.Dense(hidden_sizes[0],
            kernel_initializer='he_uniform',
            activation=activation)
        )


        for k in range(1, len(hidden_sizes)-1):
            self.model.add(
                tf.keras.layers.Dense(hidden_sizes[k],
                input_shape=(hidden_sizes[k-1],),
                kernel_initializer='he_uniform',
                activation=activation)
            )

        # No activation on the last layer !
        self.model.add(
            tf.keras.layers.Dense(1, input_shape=(hidden_sizes[-1],),
            kernel_initializer='he_uniform',
            activation=None)
        )

    def call(self, inputs):
        return self.model(inputs)