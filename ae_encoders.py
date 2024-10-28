import tensorflow as tf
from tensorflow.keras.layers import *

from typing import List


class DenseEncoder(tf.keras.Model):
    def __init__(self, hidden_layers, latent_dim, input_dim, activation):
        super().__init__()
        self.input_shape_ = (input_dim, )
        self.dense = tf.keras.Sequential(
            [Dense(k, activation=activation) for k in hidden_layers]
            + [Dense(latent_dim, activation=activation)]
        )

    def call(self, inputs, **kwargs):
        return self.dense(inputs)


class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs  # z_mean and z_log_var have shape (latent_dim, N)
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class DenseVAEEncoder(tf.keras.Model):
    def __init__(self, hidden_layers, latent_dim, input_dim, activation):
        super().__init__()

        self.dense_mean = tf.keras.Sequential(
            [Dense(input_dim, activation=activation)]
            + [Dense(k, activation=activation) for k in hidden_layers]
            + [Dense(latent_dim, activation=activation)]
        )
        self.dense_log_var = tf.keras.Sequential(
            [Dense(input_dim, activation=activation)]
            + [Dense(k, activation=activation) for k in hidden_layers]
            + [Dense(latent_dim, activation=activation)]
        )

    def call(self, inputs, **kwargs):
        z_mean = tf.clip_by_value(self.dense_mean(inputs), -1, 1)
        z_log_var = self.dense_log_var(inputs)
        z_log_var = tf.clip_by_value(z_log_var, -1, 1)
        s = Sampling()((z_mean, z_log_var))

        return z_mean, z_log_var, s


class Resnet50V2Encoder(tf.keras.Model):
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.inputs = Input(shape=input_shape, name="img")

        self.resnet = tf.keras.applications.ResNet50V2(
            include_top=False, input_shape=input_shape
        )

        """ IMPORTANT: FEATURE MODEL IS NOT TRAINABLE ! """
        self.resnet.trainable = False

        resnet_output_shape = self.resnet(self.inputs).shape

        self.final_pool = AveragePooling2D((2, 2))
        self.reshape = Reshape([-1])

        dense_layers = []
        k = resnet_output_shape[-1] // 2
        while k > latent_dim:
            dense_layers.append(Dense(k))
            dense_layers.append(LeakyReLU(0.1))
            k = k // 4
        dense_layers.append(Dense(latent_dim))
        self.dense = tf.keras.Sequential(dense_layers)

        self.calculated_output_shape = self.dense(
            self.reshape(self.final_pool(self.resnet(self.inputs)))
        ).shape

    def call(self, inputs, **kwargs):
        x = self.resnet(inputs)
        x = self.final_pool(x)
        x = self.reshape(x)
        x = self.dense(x)
        return x


class MobilenetV2Encoder(tf.keras.Model):
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.inputs = Input(shape=input_shape, name="img")

        self.mobilenet = tf.keras.applications.MobileNetV2(
            include_top=False, input_shape=input_shape
        )

        """ IMPORTANT: FEATURE MODEL IS NOT TRAINABLE ! """
        self.mobilenet.trainable = False

        mobilenet_output_shape = self.mobilenet(self.inputs).shape

        self.final_pool = AveragePooling2D((2, 2))
        self.reshape = Reshape([-1])

        dense_layers = []
        k = mobilenet_output_shape[-1] // 2
        while k > latent_dim:
            dense_layers.append(Dense(k))
            dense_layers.append(LeakyReLU(0.1))
            k = k // 4
        dense_layers.append(Dense(latent_dim))
        self.dense = tf.keras.Sequential(dense_layers)

        self.calculated_output_shape = self.dense(
            self.reshape(self.final_pool(self.mobilenet(self.inputs)))
        ).shape

    def call(self, inputs, **kwargs):
        x = self.mobilenet(inputs)
        x = self.final_pool(x)
        x = self.reshape(x)
        x = self.dense(x)
        return x


class ShallowMobilenetV2Encoder(tf.keras.Model):
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.inputs = Input(shape=input_shape, name="img")

        self.mobilenet = tf.keras.applications.MobileNetV2(
            include_top=False, input_shape=input_shape
        )

        """ IMPORTANT: FEATURE MODEL IS NOT TRAINABLE ! """
        self.mobilenet.trainable = False

        mobilenet_output_shape = self.mobilenet(self.inputs).shape

        self.final_pool = AveragePooling2D((2, 2))
        self.reshape = Reshape([-1])

        # dense_layers = []
        # k = mobilenet_output_shape[-1]//2
        # while k > latent_dim:
        #     dense_layers.append(Dense(k))
        #     dense_layers.append(LeakyReLU(0.1))
        #     k = k//4
        # dense_layers.append(Dense(latent_dim))
        self.dense = Dense(latent_dim)

        self.calculated_output_shape = self.dense(
            self.reshape(self.final_pool(self.mobilenet(self.inputs)))
        ).shape

    def call(self, inputs, **kwargs):
        x = self.mobilenet(inputs)
        x = self.final_pool(x)
        x = self.reshape(x)
        x = self.dense(x)
        return x


class Basic32x32VAE_Encoder(tf.keras.Model):
    def __init__(self, latent_dim, hidden_layers=[128, 64]):
        super().__init__()

        self.latent_dim = latent_dim

        self.conv_stack = tf.keras.Sequential(
            [
                Conv2D(10, (8, 8), strides=1, padding="same"),
                Conv2D(16, (7, 7), strides=1),
                Conv2D(24, (5, 5), strides=1, padding="same"),
                AveragePooling2D((3, 3), padding="same"),
                Conv2D(32, (3, 3), strides=1),
                Conv2D(64, (3, 3), strides=1, padding="same"),
                AveragePooling2D((3, 3), padding="same"),
                Conv2D(96, (3, 3), strides=1, padding="same"),
                Conv2D(128, (3, 3), strides=1, padding="same"),
            ]
        )

        def listcat(lists):
            rv = lists[0]
            for l in lists[1:]:
                rv = rv + l
            return rv

        self.dense_mean = tf.keras.Sequential(
            listcat(
                [
                    [Dense(hidden_layers[k]), Dropout(0.15)]
                    for k in range(len(hidden_layers))
                ]
            )
            + [Dense(self.latent_dim)]
        )

        self.dense_log_var = tf.keras.Sequential(
            listcat(
                [
                    [Dense(hidden_layers[k]), Dropout(0.15)]
                    for k in range(len(hidden_layers))
                ]
            )
            + [Dense(self.latent_dim)]
        )

    def call(self, inputs, **kwargs):
        latent = self.conv_stack(inputs)
        latent = Reshape([-1])(latent)

        z_mean = tf.clip_by_value(self.dense_mean(latent), -1, 1)

        z_log_var = self.dense_log_var(latent)
        z_log_var = tf.clip_by_value(z_log_var, -1, 1)
        s = Sampling()((z_mean, z_log_var))

        return z_mean, z_log_var, s


class Conv2DEncoder(tf.keras.Model):
    def __init__(
        self,
        latent_dim: int,
        input_dim_x: int,
        input_dim_y: int,
        channels: int,
        hidden_layers: List[int],
        activation: str,
    ):
        super().__init__()
        self.input_shape_ = (input_dim_x, input_dim_y, channels)
        conv_layers = [
                Conv2D(
                    32, (3, 3), activation=activation, padding="same"
                ),
                MaxPooling2D((2, 2)),
                Conv2D(
                    64, (3, 3), activation=activation, padding="same"
                ),
                MaxPooling2D((2, 2)),
                Conv2D(
                    128, (3, 3), activation=activation, padding="same"
                ),
                MaxPooling2D((2, 2)),
                Conv2D(
                    256, (3, 3), activation=activation, padding="same"
                ),
                MaxPooling2D((2, 2)),
                Flatten(),
        ]

        dense_layers = [Dense(layer, activation=activation) for layer in hidden_layers]
        self.model = tf.keras.Sequential(
            conv_layers + dense_layers + [Dense(latent_dim)]
        )

    def call(self, inputs):
        return self.model(inputs)


class Conv2DEncoderAvg(tf.keras.Model):
    def __init__(
        self,
        latent_dim: int,
        input_dim_x: int,
        input_dim_y: int,
        channels: int,
        hidden_layers: List[int],
        activation: str,
    ):
        super().__init__()
        self.input_shape_ = (input_dim_x, input_dim_y, channels)
        conv_layers = [
                Conv2D(
                    32, (3, 3), activation=activation, padding="same"
                ),
                AveragePooling2D((2, 2)),
                Conv2D(
                    64, (3, 3), activation=activation, padding="same"
                ),
                AveragePooling2D((2, 2)),
                Conv2D(
                    128, (3, 3), activation=activation, padding="same"
                ),
                AveragePooling2D((2, 2)),
                Conv2D(
                    256, (3, 3), activation=activation, padding="same"
                ),
                AveragePooling2D((2, 2)),
                Flatten(),
        ]

        dense_layers = [Dense(layer, activation=activation) for layer in hidden_layers]
        self.model = tf.keras.Sequential(
            conv_layers + dense_layers + [Dense(latent_dim)]
        )

    def call(self, inputs):
        return self.model(inputs)


class Conv2DEncoderLowParam(tf.keras.Model):
    def __init__(
        self,
        latent_dim: int,
        input_dim_x: int,
        input_dim_y: int,
        channels: int,
        hidden_layers: List[int],
        activation: str,
    ):
        super().__init__()
        self.input_shape_ = (input_dim_x, input_dim_y, channels)
        conv_layers = [
                Conv2D(
                    4, (3, 3), activation=activation, padding="same"
                ),
                MaxPooling2D((2, 2)),
                Conv2D(
                    8, (3, 3), activation=activation, padding="same"
                ),
                MaxPooling2D((2, 2)),
                Conv2D(
                    16, (3, 3), activation=activation, padding="same"
                ),
                MaxPooling2D((2, 2)),
                Flatten(),
        ]

        dense_layers = [Dense(layer, activation=activation) for layer in hidden_layers]
        self.model = tf.keras.Sequential(
            conv_layers + dense_layers + [Dense(latent_dim)]
        )

    def call(self, inputs):
        return self.model(inputs)


class Conv2DEncoderGeneric(tf.keras.Model):
    def __init__(
        self,
        latent_dim: int,
        input_dim_x: int,
        input_dim_y: int,
        channels: int,
        conv_filters: List[int],
        hidden_layers: List[int],
        activation: str,
        pool_size = 2,
    ):
        super().__init__()
        self.input_shape_ = (input_dim_x, input_dim_y, channels)
        conv_layers = [
            Conv2D(
                filter_num, (3, 3), activation=activation, padding="same"
            ) for filter_num in conv_filters
        ]
        
        conv_with_max_pooling = []
        for conv_layer in conv_layers:
            conv_with_max_pooling.append(conv_layer)
            conv_with_max_pooling.append(MaxPooling2D((pool_size, pool_size)))
        conv_with_max_pooling.append(Flatten())

        dense_layers = [Dense(layer, activation=activation) for layer in hidden_layers]
        self.model = tf.keras.Sequential(
            conv_with_max_pooling + dense_layers + [Dense(latent_dim)]
        )

    def call(self, inputs):
        return self.model(inputs)