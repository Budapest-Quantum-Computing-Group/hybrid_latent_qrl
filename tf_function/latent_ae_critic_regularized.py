from tensorflow.python.keras import backend as K
from tensorflow.python.keras import Model
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2

import tensorflow as tf

import logging
from typing import Callable

logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(pathname)s:%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG
)

logger = logging.getLogger(__name__)

# latent_critic with train_ae

def compute_critic_loss(
    encoder,
    decoder,
    critic,
    states,
    values,
    target,
    loss_clipping,
    latent_width,
    penalty_factor
):
    z = encoder(states)
    y_pred = critic(z)

    # critic loss
    clipped_value_loss = values + K.clip(y_pred - values, -loss_clipping, loss_clipping)

    v_loss1 = (target - clipped_value_loss) ** 2
    v_loss2 = (target - y_pred) ** 2

    c_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))

    total_loss_critic = c_loss

    s_rec = decoder( z )
    ae_loss = tf.math.reduce_mean((s_rec - states)**2)

    penalty = penalty_factor * tf.max(0.0, tf.abs(_states) - latent_width)

    total_loss_critic = total_loss_critic + ppo_agent.config.ae_coeff_critic * (ae_loss + penalty)

    return total_loss_critic


def optimize_critic(
    encoder: Model,
    decoder: Model,
    critic: Model,
    opt: OptimizerV2,
    loss_clipping: tf.Tensor,
    states: tf.Tensor,
    next_states: tf.Tensor,
    actions: tf.Tensor,
    predictions: tf.Tensor,
    values: tf.Tensor,
    next_values: tf.Tensor,
    advantages: tf.Tensor,
    target: tf.Tensor,
    latent_width,
    penalty_factor
):
    logger.info("TRACING_OPTIMIZE_CRITIC-REGULARIZED")
    with tf.GradientTape() as tape:
        loss = compute_critic_loss(
            encoder=encoder,
            decoder=decoder,
            critic=critic,
            states=states,
            values=values,
            target=target,
            loss_clipping=loss_clipping,
            latent_width=latent_width,
            penalty_factor=penalty_factor
        )
    critic_grads = tape.gradient(
        loss,
        list(critic.trainable_variables) +
        list(encoder.trainable_variables) +
        list(decoder.trainable_variables)
    )
    opt.apply_gradients(zip(
        critic_grads,
        list(critic.trainable_variables) +
        list(encoder.trainable_variables) +
        list(decoder.trainable_variables)
    ))

    return critic_grads
