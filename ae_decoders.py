import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *

from typing import List


class DenseDecoder(tf.keras.Model):
    def __init__(self, hidden_layers, latent_dim, output_dim, activation):
        super().__init__()
        self.input_shape_ = (latent_dim, )
        self.dense_stack = tf.keras.Sequential(
            [Dense(k, activation=activation) for k in hidden_layers]
            + [Dense(output_dim, activation=activation)]
        )

    def call(self, x, **kwargs):
        return self.dense_stack(x)


class BasicCNNDecoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super().__init__()

        dense = []
        k = latent_dim
        expected_dim = 1024
        while k * 2 <= expected_dim // 2:
            dense.append(Dense(k * 2))
            dense.append(LeakyReLU(0.1))
            k = k * 2
        dense.append(Dense(expected_dim))
        dense.append(LeakyReLU(0.1))

        self.upscale_0 = tf.keras.Sequential(
            dense
            + [
                Reshape([4, 4, 64]),
                Conv2DTranspose(
                    128, (3, 3), strides=1, activation="sigmoid", padding="same"
                ),
                Conv2DTranspose(
                    128, (3, 3), strides=1, activation="relu", padding="same"
                ),
                ZeroPadding2D((1, 1)),
            ]
        )

        self.upscale_1 = tf.keras.Sequential(
            [
                Conv2DTranspose(
                    128, (3, 3), strides=1, activation="sigmoid", padding="same"
                ),
                Conv2DTranspose(128, (3, 3), strides=1, padding="same"),
            ]
        )

        self.upscale_2 = tf.keras.Sequential(
            [
                Conv2DTranspose(
                    128, (3, 3), strides=2, activation="sigmoid", padding="same"
                ),
                Conv2DTranspose(128, (3, 3), strides=2, padding="same"),
            ]
        )

        self.upscale_3 = tf.keras.Sequential(
            [
                Conv2DTranspose(
                    128, (3, 3), strides=1, activation="sigmoid", padding="same"
                ),
                Conv2DTranspose(128, (3, 3), strides=1, padding="same"),
            ]
        )

        self.upscale_4 = tf.keras.Sequential(
            [
                Conv2DTranspose(
                    128, (3, 3), strides=1, activation="sigmoid", padding="same"
                ),
                Conv2DTranspose(128, (3, 3), strides=2, padding="same"),
            ]
        )

        self.upscale_5 = tf.keras.Sequential(
            [
                Conv2DTranspose(
                    128, (3, 3), strides=1, activation="sigmoid", padding="same"
                ),
                Conv2DTranspose(128, (3, 3), strides=1, padding="same"),
            ]
        )

        self.upscale_6 = tf.keras.Sequential(
            [
                Conv2DTranspose(
                    128, (3, 3), strides=1, activation="sigmoid", padding="same"
                ),
                Conv2DTranspose(128, (3, 3), strides=2, padding="same"),
                Conv2DTranspose(
                    3, (3, 3), strides=1, activation="relu", padding="same"
                ),
            ]
        )

    def call(self, x, **kwargs):
        x = self.upscale_0(x)
        x = self.upscale_1(x)
        x = LeakyReLU(0.1)(x)
        x = self.upscale_2(x)
        x = self.upscale_3(x)
        x = LeakyReLU(0.1)(x)
        x = self.upscale_4(x)
        x = self.upscale_5(x)
        x = LeakyReLU(0.1)(x)
        x = self.upscale_6(x)
        return x


class ShallowCNNDecoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super().__init__()

        dense = [Dense(1024)]

        self.upscale_0 = tf.keras.Sequential(
            dense
            + [
                Reshape([4, 4, 64]),
                Conv2DTranspose(
                    128, (3, 3), strides=1, activation="sigmoid", padding="same"
                ),
                Conv2DTranspose(
                    128, (3, 3), strides=1, activation="relu", padding="same"
                ),
                ZeroPadding2D((1, 1)),
            ]
        )

        self.upscale_1 = tf.keras.Sequential(
            [
                Conv2DTranspose(
                    128, (3, 3), strides=1, activation="sigmoid", padding="same"
                ),
                Conv2DTranspose(128, (3, 3), strides=1, padding="same"),
            ]
        )

        self.upscale_2 = tf.keras.Sequential(
            [
                Conv2DTranspose(
                    128, (3, 3), strides=2, activation="sigmoid", padding="same"
                ),
                Conv2DTranspose(128, (3, 3), strides=2, padding="same"),
            ]
        )

        self.upscale_3 = tf.keras.Sequential(
            [
                Conv2DTranspose(
                    128, (3, 3), strides=1, activation="sigmoid", padding="same"
                ),
                Conv2DTranspose(128, (3, 3), strides=1, padding="same"),
            ]
        )

        self.upscale_4 = tf.keras.Sequential(
            [
                Conv2DTranspose(
                    128, (3, 3), strides=1, activation="sigmoid", padding="same"
                ),
                Conv2DTranspose(128, (3, 3), strides=2, padding="same"),
            ]
        )

        self.upscale_5 = tf.keras.Sequential(
            [
                Conv2DTranspose(
                    128, (3, 3), strides=1, activation="sigmoid", padding="same"
                ),
                Conv2DTranspose(128, (3, 3), strides=1, padding="same"),
            ]
        )

        self.upscale_6 = tf.keras.Sequential(
            [
                Conv2DTranspose(
                    128, (3, 3), strides=1, activation="sigmoid", padding="same"
                ),
                Conv2DTranspose(128, (3, 3), strides=2, padding="same"),
                Conv2DTranspose(
                    3, (3, 3), strides=1, activation="relu", padding="same"
                ),
            ]
        )

    def call(self, x, **kwargs):
        x = self.upscale_0(x)
        x = self.upscale_1(x)
        x = LeakyReLU(0.1)(x)
        x = self.upscale_2(x)
        x = self.upscale_3(x)
        x = LeakyReLU(0.1)(x)
        x = self.upscale_4(x)
        x = self.upscale_5(x)
        x = LeakyReLU(0.1)(x)
        x = self.upscale_6(x)
        return x


class Basic32x32Decoder(tf.keras.Model):
    def __init__(self, latent_dim, hidden_layers=[64], out_channels=3):
        super().__init__()

        def listcat(lists):
            rv = lists[0]
            for l in lists[1:]:
                rv = rv + l
            return rv

        dense = listcat(
            [
                [Dense(hidden_layers[k]), Dropout(0.15)]
                for k in range(len(hidden_layers))
            ]
        ) + [Dense(1024)]

        self.upscale_0 = tf.keras.Sequential(
            dense
            + [
                Reshape([4, 4, 64]),
                Conv2DTranspose(
                    128, (3, 3), strides=1, activation="sigmoid", padding="same"
                ),
                Conv2DTranspose(
                    96, (3, 3), strides=1, activation="relu", padding="same"
                ),
                ZeroPadding2D((1, 1)),
            ]
        )

        self.upscale_1 = tf.keras.Sequential(
            [
                Conv2DTranspose(
                    64, (3, 3), strides=1, activation="sigmoid", padding="same"
                ),
                Conv2DTranspose(64, (3, 3), strides=1, padding="same"),
            ]
        )

        self.upscale_2 = tf.keras.Sequential(
            [
                Conv2DTranspose(
                    32, (5, 5), strides=1, activation="sigmoid", padding="same"
                ),
                Conv2DTranspose(32, (5, 5), strides=1, padding="same"),
            ]
        )

        self.upscale_3 = tf.keras.Sequential(
            [
                Conv2DTranspose(
                    16, (6, 6), strides=2, activation="sigmoid", padding="valid"
                ),
                Conv2DTranspose(out_channels, (7, 7), strides=2, padding="same"),
            ]
        )

    def call(self, x, **kwargs):
        x = self.upscale_0(x)
        x = self.upscale_1(x)
        x = LeakyReLU(0.1)(x)
        x = self.upscale_2(x)
        x = self.upscale_3(x)
        return x


class Conv2DDecoder(tf.keras.Model):
    def __init__(
        self,
        latent_dim: int,
        output_dim_x: int,
        output_dim_y: int,
        interpolation: str,
        channels: int,
        hidden_layers: List[int],
        activation: str,
    ):
        super().__init__()
        self.input_shape_ = (latent_dim, )
        denom = 2**4
        shape = (output_dim_x // denom, output_dim_y // denom, 256)

        reshape_layers = [
            Dense(np.prod(shape), activation=activation),
            Reshape(shape),
        ]

        dense_layers = [
            Dense(layer, activation=activation) for layer in hidden_layers
        ]

        conv_layers = [
            Conv2DTranspose(
                256, (3, 3), padding="same", activation=activation
            ),
            UpSampling2D((2, 2), interpolation=interpolation),
            Conv2DTranspose(
                128, (3, 3), padding="same", activation=activation
            ),
            UpSampling2D((2, 2), interpolation=interpolation),
            Conv2DTranspose(
                64, (3, 3), padding="same", activation=activation
            ),
            UpSampling2D((2, 2), interpolation=interpolation),
            Conv2DTranspose(
                32, (3, 3), padding="same", activation=activation
            ),
            UpSampling2D((2, 2), interpolation=interpolation),
            Conv2DTranspose(
                channels, (3, 3), padding="same", activation="sigmoid"
            )
        ]
        self.model = tf.keras.Sequential(
            dense_layers + reshape_layers + conv_layers
        )

    def call(self, inputs):
        return self.model(inputs)


class Conv2DDecoderLowParam(tf.keras.Model):
    def __init__(
        self,
        latent_dim: int,
        output_dim_x: int,
        output_dim_y: int,
        interpolation: str,
        channels: int,
        hidden_layers: List[int],
        activation: str,
    ):
        super().__init__()
        self.input_shape_ = (latent_dim, )
        denom = 2**3
        shape = (output_dim_x // denom, output_dim_y // denom, 16)

        reshape_layers = [
            Dense(np.prod(shape), activation=activation),
            Reshape(shape),
        ]

        dense_layers = [
            Dense(layer, activation=activation) for layer in hidden_layers
        ]

        conv_layers = [
            Conv2DTranspose(
                16, (3, 3), padding="same", activation=activation
            ),
            UpSampling2D((2, 2), interpolation=interpolation),
            Conv2DTranspose(
                8, (3, 3), padding="same", activation=activation
            ),
            UpSampling2D((2, 2), interpolation=interpolation),
            Conv2DTranspose(
                4, (3, 3), padding="same", activation=activation
            ),
            UpSampling2D((2, 2), interpolation=interpolation),
            Conv2DTranspose(
                channels, (3, 3), padding="same", activation="sigmoid"
            )
        ]
        self.model = tf.keras.Sequential(
            dense_layers + reshape_layers + conv_layers
        )

    def call(self, inputs):
        return self.model(inputs)


class Conv2DDecoderGeneric(tf.keras.Model):
    def __init__(
        self,
        latent_dim: int,
        output_dim_x: int,
        output_dim_y: int,
        interpolation: str,
        channels: int,
        conv_filters: List[int],
        hidden_layers: List[int],
        activation: str,
        pool_size = 2,
    ):
        super().__init__()
        self.input_shape_ = (latent_dim, )
        denom = pool_size**(len(conv_filters))
        shape = (output_dim_x // denom, output_dim_y // denom, conv_filters[0])

        reshape_layers = [
            Dense(np.prod(shape), activation=activation),
            Reshape(shape),
        ]

        dense_layers = [
            Dense(layer, activation=activation) for layer in hidden_layers
        ]

        conv_layers = [
            Conv2DTranspose(
                filter_num, (3, 3), activation=activation, padding="same"
            ) for filter_num in conv_filters
        ]

        conv_with_up_sampling = []
        for conv_layer in conv_layers:
            conv_with_up_sampling.append(conv_layer)
            conv_with_up_sampling.append(UpSampling2D((pool_size, pool_size), interpolation=interpolation))
        conv_with_up_sampling.append(
            Conv2DTranspose(
                channels, (3, 3), padding="same", activation="sigmoid"
            )
        )

        self.model = tf.keras.Sequential(
            dense_layers + reshape_layers + conv_with_up_sampling
        )

    def call(self, inputs):
        return self.model(inputs)
