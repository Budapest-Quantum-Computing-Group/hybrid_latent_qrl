import piquasso as pq
import numpy as np
from .structure import CVQNNStruct
import tensorflow as tf
import logging

logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG
)

logger = logging.getLogger(__name__)

class DisplacementEncoder(CVQNNStruct):
    def __init__(self, tfparams, n_modes, cutoff, layer_id, dtype):
        super().__init__(tfparams, n_modes, cutoff, layer_id, dtype)
        
    def structure(self, *args):
        regs = []
        gates = []
        
        for m in range(self.n_modes):
            regs.append( pq.Q(m) )
            gates.append( pq.Displacement(r=args[0][m], phi=0.0) )
                         
        return list(zip(regs, gates))

def _arctan_sqrt(x):
    if (x>0):
        return 4.0/np.pi * tf.math.pow( tf.math.abs( tf.math.atan(x) ), 1.0/3.0 )
    elif (x<0):
        return -4.0/np.pi * tf.math.pow( tf.math.abs( tf.math.atan(x) ), 1.0/3.0 )
    else:
        return x*1e-8

class DisplacementArctanEncoder(CVQNNStruct):
    def __init__(self, tfparams, n_modes, cutoff, layer_id, dtype):
        super().__init__(tfparams, n_modes, cutoff, layer_id, dtype)
        
    def structure(self, *args):
        regs = []
        gates = []
        
        for m in range(self.n_modes):
            regs.append( pq.Q(m) )
            gates.append( pq.Displacement(r=_arctan_sqrt(args[0][m]), phi=0.0) )
                         
        return list(zip(regs, gates))


class CartPoleManualEncoding(CVQNNStruct):
    def __init__(self, tfparams, n_modes, cutoff, layer_id, dtype):
        super().__init__(tfparams, n_modes, cutoff, layer_id, dtype)

    def structure(self, *args):
        regs = []
        gates = []

        regs.append(pq.Q(0))
        gates.append(pq.Displacement(r=2 * args[0][0] / 4.8, phi=0.0))  # Position

        regs.append(pq.Q(1))
        gates.append(pq.Displacement(r=2 * args[0][2] / 4.18, phi=0.0))  # Pole angle

        regs.append(pq.Q(0))
        gates.append(pq.Phaseshifter(args[0][1]))  # Velocity

        regs.append(pq.Q(1))
        gates.append(pq.Phaseshifter(args[0][3]))  # Angular velocity

        return list(zip(regs, gates))


class ManualEncodingDisplacement(CVQNNStruct):
    def __init__(self, tfparams, n_modes, cutoff, layer_id, dtype):
        super().__init__(tfparams, n_modes, cutoff, layer_id, dtype)

    def structure(self, *args):
        regs = []
        gates = []

        regs.append(pq.Q(0))
        gates.append(pq.Displacement(args[0][0], phi=np.pi / 2))

        regs.append(pq.Q(1))
        gates.append(pq.Displacement(args[0][1], phi=np.pi / 2))

        return list(zip(regs, gates))
