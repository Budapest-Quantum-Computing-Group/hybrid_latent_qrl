import piquasso as pq
import numpy as np
import tensorflow as tf

from pqcvqnn import state_processors

import logging


logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG
)

logger = logging.getLogger(__name__)


def mbexpdim(x):
    if len(x.shape)==1:
        return tf.expand_dims(
            x, axis=0, name=None
        )
    else:
        return x


class BatchedPolicy(tf.keras.Model):
    def __init__(
        self,
        layer_count: int,
        modes: int,
        cutoff: int,
        state_processor: str
    ) -> None:
        super().__init__()

        self.layer_count = layer_count
        self.modes = modes
        self.cutoff = cutoff
        self.state_processor = getattr(state_processors, state_processor)

        self.cvqnn_weights = tf.Variable(pq.cvqnn.generate_random_cvqnn_weights(layer_count=layer_count,
                                                                          d=modes), dtype=tf.float64)

        self.cvqnn_weights_checkpoint = tf.train.Checkpoint(weights=self.cvqnn_weights)

    def call(self, inputs, batch_size, calculator):
        logger.info(f"TRACING EVAL_POLICY {tf.executing_eagerly()}")
        simulator = pq.PureFockSimulator(
            self.modes,
            pq.Config(cutoff=self.cutoff, normalize=False),
            calculator=calculator
        )

        cvqnn_layers = pq.cvqnn.create_layers(self.cvqnn_weights)
        prep = []
        for i in range(batch_size):
            preparation = [pq.Vacuum()]
            inp = tf.gather(inputs, i)
            for j in range(self.modes):
                r = tf.gather(inp, j)
                preparation.append(pq.Squeezing(0.5, np.pi / 2).on_modes(j))
                preparation.append(pq.Displacement(r, 0).on_modes(j))

            prep.append(pq.Program(instructions=preparation))

        program = pq.Program(instructions=[pq.BatchPrepare(prep)] +
                             cvqnn_layers.instructions)

        final_state = simulator.execute(program).state

        return self.state_processor(final_state)


class BatchedPolicyWithReup(tf.keras.Model):
    def __init__(
        self,
        layer_count: int,
        modes: int,
        cutoff: int,
        state_processor: str,
        decorator = None
    ) -> None:
        super().__init__()

        self.layer_count = layer_count
        self.modes = modes
        self.cutoff = cutoff
        self.state_processor = getattr(state_processors, state_processor)
        self.calculator = pq.TensorflowCalculator(decorate_with=decorator)

        self.cvqnn_weights = tf.Variable(
            pq.cvqnn.generate_random_cvqnn_weights(layer_count=self.layer_count, d=self.modes),
            dtype=tf.float64
        )


    def first_encoding(self, batch: tf.Tensor, batch_size):
        prep = []
        for i in range(batch_size):
            preparation = [pq.Vacuum()]
            inp = tf.gather(batch, i)
            for j in range(self.modes):
                r = tf.gather(inp, j)
                preparation.append(pq.Squeezing(0.5, np.pi / 2).on_modes(j))
                preparation.append(pq.Displacement(r, 0).on_modes(j))

            prep.append(pq.Program(instructions=preparation))

        return prep

    def encoding(self, batch: tf.Tensor, batch_size: int):
        prep = []
        for i in range(batch_size):
            preparation = []
            inp = tf.gather(batch, i)
            for j in range(self.modes):
                r = tf.gather(inp, j)
                preparation.append(pq.Displacement(r, 0).on_modes(j))

            prep.append(pq.Program(instructions=preparation))

        return prep

    def call(self, inputs):
        batch_size = inputs.shape[0]
        config = pq.Config(cutoff=self.cutoff, normalize=False,
                           dtype=np.float64)
        simulator = pq.PureFockSimulator(
            self.modes,
            config,
            calculator=self.calculator
        )

        cvqnn_layer = pq.cvqnn.create_layers(
            mbexpdim(tf.gather(self.cvqnn_weights, 0))
        )
        prep = self.first_encoding(batch=inputs, batch_size=batch_size)

        program = pq.Program(instructions=[pq.BatchPrepare(prep)] + cvqnn_layer.instructions)

        state = simulator.execute(program).state
        prep = self.encoding(batch=inputs, batch_size=batch_size)

        for i in range(1, self.layer_count):
            cvqnn_layer = pq.cvqnn.create_layers(
                mbexpdim(tf.gather(self.cvqnn_weights, i))
            )

            program = pq.Program(instructions=[pq.BatchApply(prep)] +
                                 cvqnn_layer.instructions)

            state = simulator.execute(program, initial_state=state).state

        return self.state_processor(state)


# call with state_processor without softmax
class BatchedPolicyWithReupDense(BatchedPolicyWithReup):
    def __init__(
        self,
        layer_count: int,
        modes: int,
        cutoff: int,
        state_processor: str,
        action_space: int,
        decorator = None
    ) -> None:
        super(BatchedPolicyWithReupDense, self).__init__(layer_count, modes, cutoff, state_processor, decorator)
        self.action_space = action_space

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.modes)),
            tf.keras.layers.Dense(self.action_space, activation="softmax")
        ])

    def call(self, inputs):
        state = super().call(inputs)

        return self.model(state)
