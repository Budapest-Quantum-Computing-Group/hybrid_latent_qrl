import tensorflow as tf
def normalize_and_cast(state):
    return tf.cast( state/255.0, tf.float32 )

def normalize_and_cast_64(state):
    return tf.cast( state/255.0, tf.float64)

def normalize(state):
    return state/255.0

def identity(x):
    return x