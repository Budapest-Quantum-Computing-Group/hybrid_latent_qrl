import tensorflow as tf
import pennylane as qml
from pennylane import numpy as np
import sys
from tensorflow.keras import layers
import logging

logging.getLogger("pennylane").setLevel(logging.INFO)


def unnest_scalar(x):
    if len(x.shape) and x.shape[0] == 1:
        x = tf.squeeze(x, 0)
    return x

# this is needed !!!
def fn_attrs(**func_attrs):
    def attr_decorator(fn):
        for attr, value in func_attrs.items():
            setattr(fn, attr, value)
        return fn

    return attr_decorator

def make_circuit(feature_encoding, core_layer, n_layers, nwires, data_reuploading, measurement, measurement_wires):

    def circuit(inputs, weights):
        if not data_reuploading:
            if feature_encoding == 'angle':
                qml.AngleEmbedding(features=inputs, wires=range(nwires), rotation='X')

            elif feature_encoding == 'amplitude':
                qml.AmplitudeEmbedding(features=inputs, wires=range(nwires), normalize='True')

        for i in range(n_layers):
            if data_reuploading:
                if feature_encoding == 'angle':
                    qml.AngleEmbedding(features=inputs, wires=range(nwires), rotation='X')

                elif feature_encoding == 'amplitude':
                    raise Exception("Data reuploading is not possible with amplitude embedding, as it would require complete reinitialization of the state vector.")
                    # qml.AmplitudeEmbedding(features=inputs, wires=range(nwires), normalize='True')

            if core_layer == 'strong':
                qml.StronglyEntanglingLayers(weights=tf.reshape(weights[i], (1,nwires,3)), wires=range(nwires))

            elif core_layer == 'basic':
                qml.BasicEntanglerLayers(weights=tf.reshape(weights[i], (1,nwires)), wires=range(nwires))

        if measurement == 'expval':
            return [qml.expval(qml.PauliZ(i)) for i in measurement_wires]


    if core_layer == 'strong':
        weight_shapes = {'weights': (n_layers, nwires, 3)}

    elif core_layer == 'basic':
        weight_shapes = {'weights': (n_layers, nwires)}

    circuit = fn_attrs(weight_shapes=weight_shapes)(circuit)

    return circuit


class BasicQNNPolicy(tf.keras.Model):
    def __init__(
            self,
            n_layers=4,
            core_layer='strong',
            data_transform='identity',
            feature_encoding='angle',
            measurement='expval',
            measurement_wires = [0,1,2,3],
            use_data_reuploading=True,
            nwires=5,
            decorator=None
        ):

        super().__init__(name='BasicQNNPolicy')

        self.nwires = nwires
        self.n_layers = n_layers
        self.use_data_reuploading = use_data_reuploading
        self.core_layer = core_layer
        self.feature_encoding = feature_encoding
        self.measurement = measurement
        self.measurement_wires = measurement_wires
        self.data_transform = data_transform

        # TODO: Try lightning.qubit
        # Cuda 11.55 needed and nvidia cuquantum, lightning-gpu
        self.backend = qml.device("default.qubit", wires=self.nwires)

        f_circuit = make_circuit(
            feature_encoding = self.feature_encoding,
            core_layer = self.core_layer,
            n_layers = self.n_layers,
            nwires=self.nwires,
            data_reuploading = self.use_data_reuploading,
            measurement = self.measurement,
            measurement_wires = self.measurement_wires
        )

        circuit = qml.QNode(f_circuit, self.backend, interface='tf', diff_method='best')

        self.circuit = qml.qnn.KerasLayer(
            circuit,
            weight_shapes = f_circuit.weight_shapes,
            output_dim = 4
        )

        # Commented our as this causes an inexplainable bug in the training code
        #self.q_weights = tf.Variable(np.random.random(size = f_circuit.weight_shapes['weights']), trainable=True)


    def identity(self, x):
            return x

    def call(self, inputs, **kwargs):
        inputs = tf.cast(inputs, dtype=tf.float64)  # batch operations do not support int32

        if inputs.shape == ():
            inputs = tf.expand_dims(inputs, 0)  # () shape compatibility
            # Note: return dimension will be batched

        # check data transformations
        args = []
        for k in range(inputs.shape[-1]):
            args.append(getattr(self, self.data_transform)(unnest_scalar(inputs[..., k])))

        try:
            result = self.circuit(inputs)
        except:
            try:
                rv = []
                for i in inputs:
                    rv.append( self.circuit(i) )
                result = tf.stack(rv)
            except Exception as e:
                print(f"Failed to evaluate circuit input of shape {inputs.shape}")
                print("Exception:")
                print(e)
                sys.exit(-1)

        return tf.cast( tf.nn.softmax(result, axis=1), tf.float64)


class BasicQNNPolicy2(tf.keras.Model):
    def __init__(
            self,
            n_layers=4,
            core_layer='strong',
            feature_encoding='angle',
            measurement='expval',
            measurement_wires = [0,1,2,3],
            use_data_reuploading=True,
            nwires=4,
            decorator=None
        ):

        super().__init__(name='BasicQNNPolicyLightning')

        self.nwires = nwires
        self.n_layers = n_layers
        self.use_data_reuploading = use_data_reuploading
        self.core_layer = core_layer
        self.feature_encoding = feature_encoding
        self.measurement = measurement
        self.measurement_wires = measurement_wires

        self.backend = qml.device("default.qubit", wires=self.nwires)

        f_circuit = make_circuit(
            feature_encoding = self.feature_encoding,
            core_layer = self.core_layer,
            n_layers = self.n_layers,
            nwires=self.nwires,
            data_reuploading = self.use_data_reuploading,
            measurement = self.measurement,
            measurement_wires = self.measurement_wires
        )

        circuit = qml.QNode(f_circuit, self.backend, interface='tf', diff_method='best')

        self.circuit = qml.qnn.KerasLayer(
            circuit,
            weight_shapes = f_circuit.weight_shapes,
            output_dim = self.nwires
        )

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.nwires,)),
            self.circuit
        ])


    def call(self, inputs, **kwargs):
        inputs = tf.cast(inputs, dtype=tf.float64)  # batch operations do not support int32

        result = self.circuit(inputs)
        return tf.cast( tf.nn.softmax(result, axis=1), tf.float64)


class BasicQNNPolicyWithDense(tf.keras.Model):
    def __init__(
            self,
            n_layers=4,
            core_layer='strong',
            data_transform='identity',
            feature_encoding='angle',
            measurement='expval',
            measurement_wires = [0,1,2,3],
            use_data_reuploading=True,
            nwires=4,
            action_space=2,
            decorator=None
        ):

        super().__init__(name='BasicQNNPolicyLightning')

        self.nwires = nwires
        self.n_layers = n_layers
        self.use_data_reuploading = use_data_reuploading
        self.core_layer = core_layer
        self.feature_encoding = feature_encoding
        self.measurement = measurement
        self.measurement_wires = measurement_wires
        self.data_transform = data_transform

        self.backend = qml.device("default.qubit", wires=self.nwires)

        f_circuit = make_circuit(
            feature_encoding = self.feature_encoding,
            core_layer = self.core_layer,
            n_layers = self.n_layers,
            nwires=self.nwires,
            data_reuploading = self.use_data_reuploading,
            measurement = self.measurement,
            measurement_wires = self.measurement_wires
        )

        circuit = qml.QNode(f_circuit, self.backend, interface='tf', diff_method='best')

        self.circuit = qml.qnn.KerasLayer(
            circuit,
            weight_shapes = f_circuit.weight_shapes,
            output_dim = self.nwires
        )

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.nwires,)),
            self.circuit,
            tf.keras.layers.Dense(action_space, activation="softmax")
        ])


    def call(self, inputs, **kwargs):

        result = self.model(inputs)
        return result
