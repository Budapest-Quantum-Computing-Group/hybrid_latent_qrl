import tensorflow as tf
from tensorflow.keras.layers import *

class DiscreteActionMLP(tf.keras.Model):

    def __init__(self, obs_size, act_size, hidden_sizes, activation=None, decorator=None):
        super().__init__()

        self.input_shape_ = (obs_size, )
        
        self.model = tf.keras.Sequential()

        self.model.add(tf.keras.Input(shape=(obs_size,)))
        self.model.add(
            tf.keras.layers.Dense(hidden_sizes[0], input_shape=(obs_size,), 
            kernel_initializer=tf.random_normal_initializer(stddev=0.01), 
            activation=activation)
        )
        
        for k in range(1, len(hidden_sizes)-1):
            self.model.add(
                tf.keras.layers.Dense(hidden_sizes[k], input_shape=(hidden_sizes[k-1],),
                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                activation=activation)
            )
        
        # Softmax activation on the last layer !
        self.model.add(
            tf.keras.layers.Dense(act_size, input_shape=(hidden_sizes[-1],),
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            activation='softmax')
        )

    def call(self, inputs):
        return self.model(inputs)


class ConvImgActor(tf.keras.Model):

    def __init__(self, obs_size_x, obs_size_y, obs_size_z,
                 hidden_sizes, act_size, activation=None, decorator=None):
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
        
        self.model.add(
            tf.keras.layers.Dense(act_size, input_shape=(hidden_sizes[-1],),
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            activation="softmax")
        )

    def call(self, inputs, **kwargs):
        return self.model(inputs)

class ConvImgActorV2(tf.keras.Model):

    def __init__(self, obs_size_x, obs_size_y, obs_size_z,
                 hidden_sizes, filter_counts, act_size, activation=None, decorator=None):
        super().__init__()
        
        self.input_shape_ = (obs_size_x, obs_size_y, obs_size_z)

        self.model = tf.keras.Sequential()
        
        self.model.add(tf.keras.layers.Conv2D(
            filter_counts[0], 8, strides=4, activation=activation, padding="same"
        ))

        self.model.add(tf.keras.layers.Conv2D(
            filter_counts[1], 4, strides=2, activation=activation, padding="same"
        ))

        self.model.add(tf.keras.layers.Conv2D(
            filter_counts[2], 3, strides=2, activation=activation, padding="same"
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
        
        self.model.add(
            tf.keras.layers.Dense(act_size, input_shape=(hidden_sizes[-1],),
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            activation="softmax")
        )

    def call(self, inputs, **kwargs):
        return self.model(inputs)
