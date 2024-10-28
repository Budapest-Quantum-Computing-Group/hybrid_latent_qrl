import piquasso as pq
import numpy as np
from .structure import CVQNNStruct

class PSqueeze(CVQNNStruct):
    
    def __init__(self, tfparams, n_modes, cutoff, layer_id, dtype):
        super().__init__(tfparams, n_modes, cutoff, layer_id, dtype)
        
    def structure(self, *args):
        regs = []
        gates = []

        for m in range(self.n_modes):
            regs.append( pq.Q(m) )
            gates.append( pq.Squeezing(0.5, np.pi/2) )

        return list(zip(regs, gates))


class SqueezeZeroAngle(CVQNNStruct):
    def __init__(self, tfparams, n_modes, cutoff, layer_id, dtype):
        super().__init__(tfparams, n_modes, cutoff, layer_id, dtype)

    def structure(self, *args):
        regs = []
        gates = []

        for m in range(self.n_modes):
            regs.append(pq.Q(m))
            gates.append(pq.Squeezing(0.5, 0.0))

        return list(zip(regs, gates))
