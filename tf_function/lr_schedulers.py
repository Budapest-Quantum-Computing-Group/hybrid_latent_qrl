import numpy
import tensorflow as tf
import math

from typing import Optional

class ScoreBasedScheduler:
    def __init__(self, initial_lr: float, max_score: float) -> None:
        self.initial_lr = initial_lr
        self.max_score = max_score

    def __call__(self, score: float, episode: int):
        frac = 1.0 - (score - 1.0) / self.max_score
        return self.initial_lr * frac


class CosineDecayScheduler:
    def __init__(
        self,
        initial_lr: float,
        decay_steps: float,
        alpha: float,
        warmup_target: Optional[float] = None,
        warmup_steps: int = 0
    ) -> None:
        self.initial_lr = initial_lr
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.warmup_target = warmup_target
        self.warmup_steps = warmup_steps

    def _decay_function(self, reward, decay_steps, decay_from_lr, dtype):
        completed_fraction = reward / decay_steps
        tf_pi = tf.constant(math.pi, dtype=dtype)
        cosine_decayed = 0.5 * (1.0 + tf.cos(tf_pi * completed_fraction))
        decayed = (1 - self.alpha) * cosine_decayed + self.alpha
        return tf.multiply(decay_from_lr, decayed)

    def _warmup_function(
        self, score, warmup_steps, warmup_target, initial_lr
    ):
        completed_fraction = score / warmup_steps
        total_step_delta = warmup_target - initial_lr
        return total_step_delta * completed_fraction + initial_lr

    def __call__(self, score: float, episode: int):
        initial_lr = tf.convert_to_tensor(
            self.initial_lr, name="initial_lr"
        )
        dtype = tf.float64
        decay_steps = tf.cast(self.decay_steps, dtype)
        global_score_recomp = tf.cast(score, dtype)

        if self.warmup_target is None:
            global_score_recomp = tf.minimum(global_score_recomp, decay_steps)
            return self._decay_function(
                global_score_recomp,
                decay_steps,
                initial_lr,
                dtype,
            )

        warmup_target = tf.cast(self.warmup_target, dtype)
        warmup_steps = tf.cast(self.warmup_steps, dtype)

        global_score_recomp = tf.minimum(
            global_score_recomp, decay_steps + warmup_steps
        )

        return tf.cond(
            global_score_recomp < warmup_steps,
            lambda: self._warmup_function(
                global_score_recomp,
                warmup_steps,
                warmup_target,
                initial_lr,
            ),
            lambda: self._decay_function(
                global_score_recomp - warmup_steps,
                decay_steps,
                warmup_target,
                dtype,
            ),
        )


class InverseTimeDecay:
    def __init__(self, initial_lr: float, decay_steps: float, decay_rate: float, is_staircase: bool) -> None:
        self.initial_lr = initial_lr
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.is_staircase = is_staircase

    def __call__(self, score: float, episode: int):
        initial_lr = tf.convert_to_tensor(
            self.initial_lr, name="initial_lr"
        )
        dtype = tf.float64
        decay_steps = tf.cast(self.decay_steps, dtype)
        decay_rate = tf.cast(self.decay_rate, dtype)

        global_score_recomp = tf.cast(score, dtype)
        p = global_score_recomp / decay_steps
        if self.is_staircase:
            p = tf.floor(p)
        const = tf.cast(tf.constant(1), dtype)
        denom = tf.add(const, tf.multiply(decay_rate, p))
        return tf.divide(initial_lr, denom)


class PiecewiseConstantDecay:
    def __init__(self, initial_lr, boundaries, values):
        self.initial_lr = initial_lr
        if len(boundaries) != len(values) - 1:
                raise ValueError(
                    "The length of boundaries should be 1 less than the length of "
                    f"values. Received: boundaries={boundaries} of length "
                    f"{len(boundaries)}, and values={values} "
                    f"of length {len(values)}."
                )

        self.boundaries = boundaries
        self.values = values

    def __call__(self, score, episode):
        boundaries = tf.nest.map_structure(
            lambda x: tf.convert_to_tensor(x, dtype=tf.float64), tf.nest.flatten(self.boundaries)
        )
        values = tf.nest.map_structure(
            lambda x: tf.convert_to_tensor(x, dtype=tf.float64), tf.nest.flatten(self.values)
        )
        x_recomp = tf.convert_to_tensor(score, dtype=tf.float64)
        for i, b in enumerate(boundaries):
                b = tf.cast(b, x_recomp.dtype.base_dtype)
                boundaries[i] = b
        pred_fn_pairs = []
        pred_fn_pairs.append((x_recomp <= boundaries[0], lambda: values[0]))
        pred_fn_pairs.append(
            (x_recomp > boundaries[-1], lambda: values[-1])
        )
        for low, high, v in zip(
            boundaries[:-1], boundaries[1:], values[1:-1]
        ):
            # Need to bind v here; can do this with lambda v=v: ...
            pred = (x_recomp > low) & (x_recomp <= high)
            pred_fn_pairs.append((pred, lambda v=v: v))

        # The default isn't needed here because our conditions are mutually
        # exclusive and exhaustive, but tf.case requires it.
        default = lambda: values[0]
        return tf.case(pred_fn_pairs, default, exclusive=True)

