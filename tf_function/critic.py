from tensorflow.python.keras import backend as K
from tensorflow.python.keras import Model
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2

import tensorflow as tf

import logging

from typing import Optional

logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(pathname)s:%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG
)

logger = logging.getLogger(__name__)

# training critic only without encoder and decoder

def compute_critic_loss(
    encoder,
    decoder,
    critic,
    states,
    values,
    target,
    loss_clipping
):
    y_pred = critic(states)

    # critic loss
    clipped_value_loss = values + K.clip(y_pred - values, -loss_clipping, loss_clipping)

    v_loss1 = (target - clipped_value_loss) ** 2
    v_loss2 = (target - y_pred) ** 2

    c_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))

    return c_loss


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
    latent_width: Optional[tf.Tensor],
    penalty_factor: Optional[tf.Tensor]
):
    logger.info("TRACING_OPTIMIZE_CRITIC")
    logger.info(tf.executing_eagerly())
    with tf.GradientTape() as tape:
        loss = compute_critic_loss(
            encoder=encoder,
            decoder=decoder,
            critic=critic,
            states=states,
            values=values,
            target=target,
            loss_clipping=loss_clipping
        )
    critic_grads = tape.gradient(
        loss,
        list(critic.trainable_variables)
    )
    opt.apply_gradients(zip(
        critic_grads,
        list(critic.trainable_variables)
    ))

    return loss
