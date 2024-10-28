from tensorflow.python.keras import backend as K
from tensorflow.python.keras import Model
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2

import tensorflow as tf


import logging


logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(pathname)s:%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG
)

logger = logging.getLogger(__name__)

# latent space policy with autoencoder and decoder training

def policy_loss(
    encoder: Model,
    decoder: Model,
    policy: Model,
    states: tf.Tensor,
    actions: tf.Tensor,
    values: tf.Tensor,
    target: tf.Tensor,
    advantages: tf.Tensor,
    predictions: tf.Tensor,
    epsilon_clip: tf.Tensor,
    entropy_coeff: tf.Tensor,
    ae_coeff_policy: tf.Tensor,
    latent_width: tf.Tensor,
    penalty_factor: tf.Tensor
):
    _states = encoder(states)

    predicted_probs = policy(
        inputs=_states
    )

    # ppo loss
    prob = actions * predicted_probs
    old_prob = actions * predictions

    prob     = K.clip(prob, 1e-10, 1.0)
    old_prob = K.clip(old_prob, 1e-10, 1.0)

    ratio = K.exp(K.log(prob) - K.log(old_prob))

    p1 = ratio * advantages
    p2 = K.clip(ratio, min_value=1 - epsilon_clip, max_value=1 + epsilon_clip) * advantages

    actor_loss = -K.mean(K.minimum(p1, p2))

    entropy = -(predicted_probs * K.log(predicted_probs + 1e-10))
    entropy = entropy_coeff * K.mean(entropy)

    ppo_loss = actor_loss - entropy

    total_loss_policy = ppo_loss

    s_rec = decoder(_states)
    ae_loss = tf.math.reduce_mean((s_rec - states)**2)

    penalty = penalty_factor * tf.reduce_sum(tf.math.maximum(0.0, tf.abs(_states) - latent_width))

    ae_loss += penalty

    total_loss_policy = total_loss_policy + ae_coeff_policy * ae_loss

    return total_loss_policy


# @tf.function(jit_compile=False)
def optimize_policy(
    encoder: Model,
    decoder: Model,
    policy: Model,
    opt: OptimizerV2,
    states: tf.Tensor,
    next_states: tf.Tensor,
    actions: tf.Tensor,
    predictions: tf.Tensor,
    values: tf.Tensor,
    next_values: tf.Tensor,
    advantages: tf.Tensor,
    target: tf.Tensor,
    epsilon_clip: tf.Tensor,
    entropy_coeff: tf.Tensor,
    ae_coeff_policy: tf.Tensor,
    latent_width: tf.Tensor,
    penalty_factor: tf.Tensor
):
    logger.info("TRACING LATENT-AE-OPTIMIZE-REGULARIZED-POLICY")
    with tf.GradientTape() as tape:
        loss = policy_loss(
            encoder=encoder,
            decoder=decoder,
            policy=policy,
            states=states,
            actions=actions,
            values=values,
            target=target,
            advantages=advantages,
            predictions=predictions,
            epsilon_clip=epsilon_clip,
            entropy_coeff=entropy_coeff,
            ae_coeff_policy=ae_coeff_policy,
            latent_width=latent_width,
            penalty_factor=penalty_factor
        )

    policy_grads = tape.gradient(loss,
                                 list(policy.trainable_variables) +
                                 list(encoder.trainable_variables) +
                                 list(decoder.trainable_variables))

    opt.apply_gradients(zip(policy_grads,
                           list(policy.trainable_variables) +
                           list(encoder.trainable_variables) +
                           list(decoder.trainable_variables)))

    return policy_grads
