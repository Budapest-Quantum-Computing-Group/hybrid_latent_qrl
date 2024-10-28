from abc import ABCMeta, abstractmethod

class CVQNNStruct:
    def __init__(self, tfparams, n_modes, cutoff, layer_id, dtype):
        self.tfparams = tfparams
        self.n_modes = n_modes
        self.cutoff = cutoff
        self.layer_id = layer_id
        self.dtype = dtype
        pass
    
    @abstractmethod
    def structure(self, *args):
        # must return list(zip(regs, gates))
        pass