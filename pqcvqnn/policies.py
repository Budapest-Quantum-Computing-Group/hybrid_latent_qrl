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

logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG
)

logger = logging.getLogger(__name__)

class CVQNNPolicy(tf.keras.Model):
    def __init__(
            self,
            
            n_layers = 2,
            n_modes = 3,
            cutoff = 10,

            transform='identity',
            state_initializer='PSqueeze',
            feature_encoder='DisplacementEncoder',
            core_layer='ThreeModeCoreLayer',
            state_processor='two_mean_x',
            
            use_data_reuploading=True,

            dtype = np.float64
        ):

        super().__init__(name='CVQNNPolicy')

        self.n_layers = n_layers
        self.n_modes = n_modes
        self.cutoff = cutoff
        self.use_data_reuploading = use_data_reuploading

        self.simulator = pq.PureFockSimulator(
            d=self.n_modes, config=pq.Config(cutoff=self.cutoff, dtype=dtype),
            calculator=pq.TensorFlowCalculator()
        )

        # Stores weight parameters for TF weights
        self.tfparams = {}

        try:
            self.transform = getattr(transforms, transform)
        except:
            logger.fatal(f"Could not load transform {transform} from transforms.")
            sys.exit(128)

        try:
            self.state_initializer = getattr(initializers, state_initializer)(self.tfparams, self.n_modes, self.cutoff, layer_id=0, dtype=dtype)
        except:
            logger.fatal(f"Could not load state_initializer {state_initializer} from initializers.")
            sys.exit(128)

        if self.use_data_reuploading:
            self.feature_encoders = []

            for L in range(self.n_layers):
                try:
                    self.feature_encoders.append(
                        getattr(encodings, feature_encoder)(self.tfparams, self.n_modes, self.cutoff, layer_id=L, dtype=dtype)
                    )
                except:
                    logger.fatal(f"Could not load feature_encoder {feature_encoder} from encodings.")
                    sys.exit(128)
        else:
            try:
                self.feature_encoder = getattr(encodings, feature_encoder)(self.tfparams, self.n_modes, self.cutoff, layer_id=0, dtype=dtype)
            except:
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
                    getattr(core_layers, core_layer)(self.tfparams, self.n_modes, self.cutoff, layer_id=L, dtype=dtype)
                )
            except:
                logger.fatal(f"Could not load core_layer {core_layer} from core_layers.")
                sys.exit(128)
        
    def get_program(self, inputs):
        

        with pq.Program() as program:
            
            pq.Q(all) | pq.StateVector((0,)*self.n_modes) # n-mode vacuum
            
            for reg, gate in self.state_initializer.structure():
                reg | gate
                    
            if not self.use_data_reuploading:
                for reg, gate in self.feature_encoder.structure(inputs):
                    reg | gate
                    
            for L in range(self.n_layers):
                
                if self.use_data_reuploading:
                    for reg, gate in self.feature_encoders[L].structure(inputs):
                        reg | gate
                         
                for reg, gate in self.core_layers[L].structure():
                    reg | gate
                
                    
            return program

    def call(self, inputs, **kwargs):
        
        outputs = []
        for inp in inputs:
            
            program = self.get_program(inp)
            
            state = self.simulator.execute(program).state
            
            outputs.append(
                self.state_processor(state)
            )
            
        return tf.stack(outputs)


class CVQNNPolicyManualEncodingCartPole(CVQNNPolicy):
    def __init__(
        self,
        n_layers=2,
        n_modes=3,
        cutoff=10,
        transform='identity',
        state_initializer='PSqueeze',
        feature_encoder='DisplacementEncoder',
        core_layer='ThreeModeCoreLayer',
        state_processor='two_mean_x',
        use_data_reuploading=True,
        dtype=np.float64
    ):
        super().__init__(n_layers, n_modes, cutoff, transform, state_initializer,
                         feature_encoder, core_layer, state_processor,
                         use_data_reuploading, dtype)

        if not self.use_data_reuploading:
            assert isinstance(self.feature_encoder, encodings.CartPoleManualEncoding)
        else:
            for encoder in self.feature_encoders:
                assert isinstance(encoder, encodings.CartPoleManualEncoding)

    def get_program(self, inputs):
        assert self.n_modes == 2
        with pq.Program() as program:
            pq.Q(all) | pq.StateVector((0,)*self.n_modes) # n-mode vacuum
            
            for reg, gate in self.state_initializer.structure():
                reg | gate
    
            if not self.use_data_reuploading:
                self.feature_encoder.structure(inputs)
                    
            for L in range(self.n_layers):
                
                if self.use_data_reuploading:
                    self.feature_encoders[L].structure(inputs)
                        
                for reg, gate in self.core_layers[L].structure():
                    reg | gate
                
                    
            return program
