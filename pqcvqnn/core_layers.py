import tensorflow as tf
import numpy as np
import piquasso as pq
from .structure import CVQNNStruct

class TwoModeCoreLayer(CVQNNStruct):
    
    def __init__(self, tfparams, n_modes, cutoff, layer_id, dtype):
        super().__init__(tfparams, n_modes, cutoff, layer_id, dtype)
        assert n_modes==2, "expected 2 modes for `TwoModeCoreLayer`"
        
        self.tfparams[f"BS1_theta[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        self.tfparams[f"BS1_phi[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        
        for m in range(self.n_modes):
            self.tfparams[f"r[{layer_id}][{m}]"] = tf.Variable(np.random.normal(0.0, 0.05), dtype=self.dtype)
            self.tfparams[f"phi1[{layer_id}][{m}]"] = tf.Variable(np.random.uniform(0.0, np.pi), dtype=self.dtype)
        
        self.tfparams[f"BS2_theta[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        self.tfparams[f"BS2_phi[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        
        for m in range(self.n_modes):
            self.tfparams[f"s[{layer_id}][{m}]"] = tf.Variable(np.random.normal(0.0, 0.05), dtype=self.dtype)
            self.tfparams[f"phi2[{layer_id}][{m}]"] = tf.Variable(np.random.uniform(0.0, np.pi), dtype=self.dtype)
            self.tfparams[f"k[{layer_id}][{m}]"] = tf.Variable(np.random.uniform(0.0, 0.2), dtype=self.dtype)
        
        # for key in self.tfparams.keys():
        #     self.tfparams[key] = tf.cast(self.tfparams[key], tf.float64)
        
    def structure(self, *args):
        regs = []
        gates = []

        for m in range(self.n_modes):
            regs.append( pq.Q(m) )
            gates.append( pq.Displacement(r=self.tfparams[f"r[{self.layer_id}][{m}]"], phi=0.0) )
        
        regs.append( pq.Q(0,1) )
        gates.append( pq.Beamsplitter(theta=self.tfparams[f"BS1_theta[{self.layer_id}]"], phi=self.tfparams[f"BS1_phi[{self.layer_id}]"]))
        

        for m in range(self.n_modes):
            regs.append( pq.Q(m) )
            gates.append( pq.Phaseshifter(phi=self.tfparams[f"phi1[{self.layer_id}][{m}]"]) )
        
        regs.append( pq.Q(0,1) )
        gates.append( pq.Beamsplitter(theta=self.tfparams[f"BS2_theta[{self.layer_id}]"], phi=self.tfparams[f"BS2_phi[{self.layer_id}]"]))


        for m in range(self.n_modes):
            regs.append( pq.Q(m) )
            gates.append( pq.Phaseshifter(phi=self.tfparams[f"phi2[{self.layer_id}][{m}]"]) )

        
        for m in range(self.n_modes):
            regs.append( pq.Q(m) )
            gates.append( pq.Squeezing(r=self.tfparams[f"s[{self.layer_id}][{m}]"], phi=0.0) )
            
            regs.append( pq.Q(m) )
            gates.append( pq.Kerr(xi=self.tfparams[f"k[{self.layer_id}][{m}]"]) )

        return list(zip(regs, gates))


class TwoModeCoreLayer2(CVQNNStruct):
    
    def __init__(self, tfparams, n_modes, cutoff, layer_id, dtype):
        super().__init__(tfparams, n_modes, cutoff, layer_id, dtype)
        assert n_modes==2, "expected 2 modes for `TwoModeCoreLayer`"
        
        self.tfparams[f"BS1_theta[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        self.tfparams[f"BS1_phi[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        
        for m in range(self.n_modes):
            self.tfparams[f"r[{layer_id}][{m}]"] = tf.Variable(np.random.normal(0.0, 0.05), dtype=self.dtype)
            self.tfparams[f"phi1[{layer_id}][{m}]"] = tf.Variable(np.random.uniform(0.0, np.pi), dtype=self.dtype)
        
        self.tfparams[f"BS2_theta[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        self.tfparams[f"BS2_phi[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        
        for m in range(self.n_modes):
            self.tfparams[f"s[{layer_id}][{m}]"] = tf.Variable(np.random.normal(0.0, 0.05), dtype=self.dtype)
            self.tfparams[f"phi2[{layer_id}][{m}]"] = tf.Variable(np.random.uniform(0.0, np.pi), dtype=self.dtype)
            self.tfparams[f"k[{layer_id}][{m}]"] = tf.Variable(np.random.uniform(0.0, 0.2), dtype=self.dtype)
        
        # for key in self.tfparams.keys():
        #     self.tfparams[key] = tf.cast(self.tfparams[key], tf.float64)
        
    def structure(self, *args):
        regs = []
        gates = []
        
        regs.append( pq.Q(0,1) )
        gates.append( pq.Beamsplitter(theta=self.tfparams[f"BS1_theta[{self.layer_id}]"], phi=self.tfparams[f"BS1_phi[{self.layer_id}]"]))
        
        for m in range(self.n_modes):
            regs.append( pq.Q(m) )
            gates.append( pq.Phaseshifter(phi=self.tfparams[f"phi1[{self.layer_id}][{m}]"]) )

        for m in range(self.n_modes):
            regs.append( pq.Q(m) )
            gates.append( pq.Squeezing(r=self.tfparams[f"s[{self.layer_id}][{m}]"], phi=0.0) )
        
        regs.append( pq.Q(0,1) )
        gates.append( pq.Beamsplitter(theta=self.tfparams[f"BS2_theta[{self.layer_id}]"], phi=self.tfparams[f"BS2_phi[{self.layer_id}]"]))

        for m in range(self.n_modes):
            regs.append( pq.Q(m) )
            gates.append( pq.Phaseshifter(phi=self.tfparams[f"phi2[{self.layer_id}][{m}]"]) )

        
        for m in range(self.n_modes):
            regs.append( pq.Q(m) )
            gates.append( pq.Displacement(r=self.tfparams[f"r[{self.layer_id}][{m}]"], phi=0.0) )
            
            regs.append( pq.Q(m) )
            gates.append( pq.Kerr(xi=self.tfparams[f"k[{self.layer_id}][{m}]"]) )

        return list(zip(regs, gates))

class ThreeModeCoreLayer(CVQNNStruct):
    
    def __init__(self, tfparams, n_modes, cutoff, layer_id, dtype):
        super().__init__(tfparams, n_modes, cutoff, layer_id, dtype)
        assert n_modes==3, "expected 3 modes for `ThreeModeCoreLayer`"
        
        self.tfparams[f"BS1_theta[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        self.tfparams[f"BS1_phi[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        
        self.tfparams[f"BS2_theta[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        self.tfparams[f"BS2_phi[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        
        self.tfparams[f"BS3_theta[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        self.tfparams[f"BS3_phi[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        
        for m in range(self.n_modes):
            self.tfparams[f"r[{layer_id}][{m}]"] = tf.Variable(np.random.normal(0.0, 0.05), dtype=self.dtype)
            self.tfparams[f"phi1[{layer_id}][{m}]"] = tf.Variable(np.random.uniform(0.0, np.pi), dtype=self.dtype)
        
        self.tfparams[f"BS4_theta[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        self.tfparams[f"BS4_phi[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        
        self.tfparams[f"BS5_theta[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        self.tfparams[f"BS5_phi[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        
        self.tfparams[f"BS6_theta[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        self.tfparams[f"BS6_phi[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        
        
        for m in range(self.n_modes):
            self.tfparams[f"s[{layer_id}][{m}]"] = tf.Variable(np.random.normal(0.0, 0.05), dtype=self.dtype)
            self.tfparams[f"phi2[{layer_id}][{m}]"] = tf.Variable(np.random.uniform(0.0, np.pi), dtype=self.dtype)
            self.tfparams[f"k[{layer_id}][{m}]"] = tf.Variable(np.random.uniform(0.0, 0.2), dtype=self.dtype)
        
        # for key in self.tfparams.keys():
        #     self.tfparams[key] = tf.cast(self.tfparams[key], tf.float64)
            
    def structure(self, *args):
        regs = []
        gates = []

        for m in range(self.n_modes):
            regs.append( pq.Q(m) )
            gates.append( pq.Displacement(r=self.tfparams[f"r[{self.layer_id}][{m}]"], phi=0.0) )
        
        regs.append( pq.Q(0,1) )
        gates.append( pq.Beamsplitter(theta=self.tfparams[f"BS1_theta[{self.layer_id}]"], phi=self.tfparams[f"BS1_phi[{self.layer_id}]"]))
        
        regs.append( pq.Q(1,2) )
        gates.append( pq.Beamsplitter(theta=self.tfparams[f"BS2_theta[{self.layer_id}]"], phi=self.tfparams[f"BS2_phi[{self.layer_id}]"]))

        regs.append( pq.Q(0,1) )
        gates.append( pq.Beamsplitter(theta=self.tfparams[f"BS3_theta[{self.layer_id}]"], phi=self.tfparams[f"BS3_phi[{self.layer_id}]"]))

        for m in range(self.n_modes):
            regs.append( pq.Q(m) )
            gates.append( pq.Phaseshifter(phi=self.tfparams[f"phi1[{self.layer_id}][{m}]"]) )
        
        regs.append( pq.Q(1,2) )
        gates.append( pq.Beamsplitter(theta=self.tfparams[f"BS4_theta[{self.layer_id}]"], phi=self.tfparams[f"BS4_phi[{self.layer_id}]"]))
        
        regs.append( pq.Q(0,1) )
        gates.append( pq.Beamsplitter(theta=self.tfparams[f"BS5_theta[{self.layer_id}]"], phi=self.tfparams[f"BS5_phi[{self.layer_id}]"]))

        regs.append( pq.Q(1,2) )
        gates.append( pq.Beamsplitter(theta=self.tfparams[f"BS6_theta[{self.layer_id}]"], phi=self.tfparams[f"BS6_phi[{self.layer_id}]"]))


        for m in range(self.n_modes):
            regs.append( pq.Q(m) )
            gates.append( pq.Phaseshifter(phi=self.tfparams[f"phi2[{self.layer_id}][{m}]"]) )
        
        for m in range(self.n_modes):
            regs.append( pq.Q(m) )
            gates.append( pq.Squeezing(r=self.tfparams[f"s[{self.layer_id}][{m}]"], phi=0.0) )
            
            regs.append( pq.Q(m) )
            gates.append( pq.Kerr(xi=self.tfparams[f"k[{self.layer_id}][{m}]"]) )

        return list(zip(regs, gates))
        

class FourModeCoreLayer(CVQNNStruct):
    
    def __init__(self, tfparams, n_modes, cutoff, layer_id, dtype):
        super().__init__(tfparams, n_modes, cutoff, layer_id, dtype)
        assert n_modes==4, "expected 4 modes for `FourModeCoreLayer`"
        
        self.tfparams[f"BS1_theta[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        self.tfparams[f"BS1_phi[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        
        self.tfparams[f"BS2_theta[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        self.tfparams[f"BS2_phi[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        
        self.tfparams[f"BS3_theta[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        self.tfparams[f"BS3_phi[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)

        self.tfparams[f"BS4_theta[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        self.tfparams[f"BS4_phi[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)

        self.tfparams[f"BS5_theta[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        self.tfparams[f"BS5_phi[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        
        for m in range(self.n_modes):
            self.tfparams[f"r[{layer_id}][{m}]"] = tf.Variable(np.random.normal(0.0, 0.05), dtype=self.dtype)
            self.tfparams[f"phi1[{layer_id}][{m}]"] = tf.Variable(np.random.uniform(0.0, np.pi), dtype=self.dtype)
        
        self.tfparams[f"BS6_theta[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        self.tfparams[f"BS6_phi[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)

        self.tfparams[f"BS7_theta[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        self.tfparams[f"BS7_phi[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)

        self.tfparams[f"BS8_theta[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        self.tfparams[f"BS8_phi[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)

        self.tfparams[f"BS9_theta[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        self.tfparams[f"BS9_phi[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)

        self.tfparams[f"BS10_theta[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        self.tfparams[f"BS10_phi[{layer_id}]"] = tf.Variable(np.random.normal(0.0, 1.0), dtype=self.dtype)
        
        for m in range(self.n_modes):
            self.tfparams[f"s[{layer_id}][{m}]"] = tf.Variable(np.random.normal(0.0, 0.05), dtype=self.dtype)
            self.tfparams[f"phi2[{layer_id}][{m}]"] = tf.Variable(np.random.uniform(0.0, np.pi), dtype=self.dtype)
            self.tfparams[f"k[{layer_id}][{m}]"] = tf.Variable(np.random.uniform(0.0, 0.2), dtype=self.dtype)
        
        # for key in self.tfparams.keys():
        #     self.tfparams[key] = tf.cast(self.tfparams[key], tf.float64)
            
    def structure(self, *args):
        regs = []
        gates = []

        for m in range(self.n_modes):
            regs.append( pq.Q(m) )
            gates.append( pq.Displacement(r=self.tfparams[f"r[{self.layer_id}][{m}]"], phi=0.0) )
        
        regs.append( pq.Q(0,1) )
        gates.append( pq.Beamsplitter(theta=self.tfparams[f"BS1_theta[{self.layer_id}]"], phi=self.tfparams[f"BS1_phi[{self.layer_id}]"]))
        
        regs.append( pq.Q(1,2) )
        gates.append( pq.Beamsplitter(theta=self.tfparams[f"BS2_theta[{self.layer_id}]"], phi=self.tfparams[f"BS2_phi[{self.layer_id}]"]))

        regs.append( pq.Q(2,3) )
        gates.append( pq.Beamsplitter(theta=self.tfparams[f"BS3_theta[{self.layer_id}]"], phi=self.tfparams[f"BS3_phi[{self.layer_id}]"]))

        regs.append( pq.Q(1,2) )
        gates.append( pq.Beamsplitter(theta=self.tfparams[f"BS4_theta[{self.layer_id}]"], phi=self.tfparams[f"BS4_phi[{self.layer_id}]"]))

        regs.append( pq.Q(0,1) )
        gates.append( pq.Beamsplitter(theta=self.tfparams[f"BS5_theta[{self.layer_id}]"], phi=self.tfparams[f"BS5_phi[{self.layer_id}]"]))

        for m in range(self.n_modes):
            regs.append( pq.Q(m) )
            gates.append( pq.Phaseshifter(phi=self.tfparams[f"phi1[{self.layer_id}][{m}]"]) )
        
        
        regs.append( pq.Q(2,3) )
        gates.append( pq.Beamsplitter(theta=self.tfparams[f"BS6_theta[{self.layer_id}]"], phi=self.tfparams[f"BS6_phi[{self.layer_id}]"]))

        regs.append( pq.Q(1,2) )
        gates.append( pq.Beamsplitter(theta=self.tfparams[f"BS7_theta[{self.layer_id}]"], phi=self.tfparams[f"BS7_phi[{self.layer_id}]"]))

        regs.append( pq.Q(0,1) )
        gates.append( pq.Beamsplitter(theta=self.tfparams[f"BS8_theta[{self.layer_id}]"], phi=self.tfparams[f"BS8_phi[{self.layer_id}]"]))

        regs.append( pq.Q(1,2) )
        gates.append( pq.Beamsplitter(theta=self.tfparams[f"BS9_theta[{self.layer_id}]"], phi=self.tfparams[f"BS9_phi[{self.layer_id}]"]))


        regs.append( pq.Q(2,3) )
        gates.append( pq.Beamsplitter(theta=self.tfparams[f"BS10_theta[{self.layer_id}]"], phi=self.tfparams[f"BS10_phi[{self.layer_id}]"]))


        for m in range(self.n_modes):
            
            regs.append( pq.Q(m) )
            gates.append( pq.Phaseshifter(phi=self.tfparams[f"phi2[{self.layer_id}][{m}]"]) )
        
        for m in range(self.n_modes):
            regs.append( pq.Q(m) )
            gates.append( pq.Squeezing(r=self.tfparams[f"s[{self.layer_id}][{m}]"], phi=0.0) )
            
            regs.append( pq.Q(m) )
            gates.append( pq.Kerr(xi=self.tfparams[f"k[{self.layer_id}][{m}]"]) )

        return list(zip(regs, gates))