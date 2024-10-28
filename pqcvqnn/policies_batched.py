import piquasso as pq
import numpy as np
import tensorflow as tf 

from . import structure
from . import initializers
from . import transforms
from . import encodings
from . import core_layers
from . import state_processors

import sys

import logging

from typing import Iterable


logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG
)

logger = logging.getLogger(__name__)

class CVQNNPolicyBatched(tf.keras.Model):
    def __init__(
            self,
            
            n_layers = 2,
            n_modes = 3,
            cutoff = 10,

            transform='identity',
            state_initializer='PSqueeze',
            feature_encoder='DisplacementEncoder',
            core_layer='ThreeModeCoreLayer',
            state_processor='twomode_mean_p_batched_softmax',
            
            use_data_reuploading=True,

            dtype_np = np.float64,
            dtype_tf = tf.float64
        ):

        super().__init__(name='CVQNNPolicyBatched')

        self.n_layers = n_layers
        self.n_modes = n_modes
        self.cutoff = cutoff
        self.use_data_reuploading = use_data_reuploading

      

        self.dtype_np = dtype_np 
        self.dtype_tf = dtype_tf 

        self.simulator = pq.PureFockSimulator(
            d=self.n_modes, config=pq.Config(cutoff=self.cutoff, dtype=self.dtype_np),
            calculator=pq.TensorflowCalculator()
        )

        # Stores weight parameters for TF weights
        self.tfparams = {}

        try:
            self.transform = getattr(transforms, transform)
        except:
            logger.fatal(f"Could not load transform {transform} from transforms.")
            sys.exit(128)

        try:
            self.state_initializer = getattr(initializers, state_initializer)(self.tfparams, self.n_modes, self.cutoff, layer_id=0, dtype=self.dtype_tf)
        except:
            logger.fatal(f"Could not load state_initializer {state_initializer} from initializers.")
            sys.exit(128)

        try:
            self.feature_encoder = getattr(encodings, feature_encoder)(self.tfparams, self.n_modes, self.cutoff, layer_id=0, dtype=self.dtype_tf)
        except Exception as e:
            logger.fatal(f"Could not load feature_encoder {feature_encoder} from encodings.")
            sys.exit(128)

        try:
            self.state_processor = getattr(state_processors, state_processor)
        except:
            logger.fatal(f"Could not load state_processor {state_processor} from state_processors.")
            sys.exit(128)
        

        self.core_layers = []
        
        for L in range(self.n_layers):

            try:
                self.core_layers.append(
                    getattr(core_layers, core_layer)(self.tfparams, self.n_modes, self.cutoff, layer_id=L, dtype=self.dtype_tf)
                )
            except:
                logger.fatal(f"Could not load core_layer {core_layer} from core_layers.")
                sys.exit(128)

    def get_first_encoding(self, ibatch):
        
        first_encoding = []
        
        for inp in ibatch:
            with pq.Program() as encoding:
                pq.Q(all) | pq.StateVector((0,)*self.n_modes) # n-mode vacuum
                
                for reg, gate in self.state_initializer.structure():
                    reg | gate
                    
                for reg, gate in self.feature_encoder.structure(inp):
                    reg | gate
                    
            first_encoding.append(encoding)
            
        return first_encoding
    
    def get_reup_encoding(self, ibatch):
        reup_encoding = []
        
        for inp in ibatch:
            with pq.Program() as encoding:
                for reg, gate in self.feature_encoder.structure(inp):
                    reg | gate
                    
            reup_encoding.append(encoding)

        return reup_encoding
        
    def get_program(self, ibatch):
        
        #assert not self.use_data_reuploading, "Data re-uploading currently not supported in batch mode."
        first_encoding = self.get_first_encoding(ibatch)
        
        if self.use_data_reuploading:
            reup_encoding = self.get_reup_encoding(ibatch)

        with pq.Program() as program:
            
            pq.Q() | pq.BatchPrepare(first_encoding)
            
            for L in range(self.n_layers):
                
                if self.use_data_reuploading:
                    pq.Q() | pq.BatchApply(reup_encoding)
                         
                for reg, gate in self.core_layers[L].structure():
                    reg | gate
                
                    
            return program

    def call(self, inputs, **kwargs):
        
            
        program = self.get_program(inputs)

        state = self.simulator.execute(program).state

        outputs = self.state_processor(state)
            
        return outputs


class CVQNNPolicyManualEncodingBatched(CVQNNPolicyBatched):
    def __init__(
            self,
            feature_indices: Iterable[int],
            n_layers=2,
            n_modes=3,
            cutoff=10,
            transform='identity',
            state_initializer='PSqueeze',
            feature_encoder='DisplacementEncoder',
            core_layer='ThreeModeCoreLayer',
            state_processor='twomode_mean_p_batched_softmax',
            use_data_reuploading=True,
            dtype_np=np.float64,
            dtype_tf=tf.float64,
            use_arctan: bool=False
        ):
        super().__init__(n_layers, n_modes, cutoff, transform,
                         state_initializer, feature_encoder,
                         core_layer, state_processor,
                         use_data_reuploading, dtype_np, dtype_tf)
        self.use_arctan = use_arctan
        self.feature_indices = feature_indices

    def encode(self, batch):
        # batch = tf.gather(batch, self.feature_indices)
        y1 = 2 * (2.0 / np.pi) * tf.math.atan(batch)
        encoded = tf.where(
            tf.less(batch, 0.0),
            -tf.math.pow(-y1, 1 / 3),
            tf.where(
                tf.greater(batch, 0.0),
                tf.math.pow(y1, 1 / 3),
                tf.zeros_like(batch)
            )
        )
        return encoded
    
    # def _transform(self, x):
    #         return tf.gather(x, self.feature_indices)
    
    # inputs: Tensor with shape (x, y)
    # where x is the number of latent_qrl_envs
    # and y is the observation space
    def call(self, inputs, **kwargs):
        # inputs = inputs.map(self._transform)
        inputs = tf.gather(inputs, self.feature_indices, axis=1)
        if self.use_arctan:
            inputs = self.encode(inputs)
            # inputs = inputs.map(self.encode)
        return super().call(inputs)        
