import tensorflow as tf
import piquasso as pq
import numpy as np

def twomode_mean_x(state):

    m0, _ = state.quadratures_mean_variance(modes=[0])
    m1, _ = state.quadratures_mean_variance(modes=[1])

    return tf.stack([m0,m1])

def twomode_mean_x_softmax(state):

    m0, _ = state.quadratures_mean_variance(modes=[0])
    m1, _ = state.quadratures_mean_variance(modes=[1])

    return tf.nn.softmax(tf.stack([m0,m1]), axis=0)

def twomode_mean_p_batched(state):

    m0 = state.mean_position(mode=0)
    m1 = state.mean_position(mode=1)

    # Output shape will be (batch_size,2)
    return tf.stack([m0,m1], axis=1)

def twomode_mean_p_batched_softmax(state):

    m0 = state.mean_position(mode=0)
    m1 = state.mean_position(mode=1)

    # Output shape will be (batch_size,2)
    return tf.nn.softmax(tf.stack([m0,m1], axis=1), axis=1)


def threemode_mean_p_batched(state):

    m0 = state.mean_position(mode=0)
    m1 = state.mean_position(mode=1)
    m2 = state.mean_position(mode=2)


    # Output shape will be (batch_size,3)
    return tf.stack([m0,m1,m2], axis=1)

def threemode_mean_p_batched_softmax(state):

    m0 = state.mean_position(mode=0)
    m1 = state.mean_position(mode=1)
    m2 = state.mean_position(mode=2)

    # Output shape will be (batch_size,3)
    return tf.nn.softmax(tf.stack([m0,m1,m2], axis=1), axis=1)


def fourmode_mean_p_batched(state):

    m0 = state.mean_position(mode=0)
    m1 = state.mean_position(mode=1)
    m2 = state.mean_position(mode=2)
    m3 = state.mean_position(mode=3)


    # Output shape will be (batch_size,4)
    return tf.stack([m0,m1,m2,m3], axis=1)

def fourmode_mean_p_batched_softmax(state):

    m0 = state.mean_position(mode=0)
    m1 = state.mean_position(mode=1)
    m2 = state.mean_position(mode=2)
    m3 = state.mean_position(mode=3)

    # Output shape will be (batch_size,4)
    return tf.nn.softmax(tf.stack([m0,m1,m2,m3], axis=1), axis=1)

def sixmode_mean_p_batched(state):

    m0 = state.mean_position(mode=0)
    m1 = state.mean_position(mode=1)
    m2 = state.mean_position(mode=2)
    m3 = state.mean_position(mode=3)
    m4 = state.mean_position(mode=4)
    m5 = state.mean_position(mode=5)

    return tf.stack([m0,m1,m2,m3,m4,m5], axis=1)

def eightmode_mean_p_batched_softmax(state):

    m0 = state.mean_position(mode=0)
    m1 = state.mean_position(mode=1)
    m2 = state.mean_position(mode=2)
    m3 = state.mean_position(mode=3)
    m4 = state.mean_position(mode=4)
    m5 = state.mean_position(mode=5)
    m6 = state.mean_position(mode=6)
    m7 = state.mean_position(mode=7)

    return tf.nn.softmax(tf.stack([m0,m1,m2,m3,m4,m5,m6,m7], axis=1), axis=1)


def eightmode_mean_p_batched(state):

    m0 = state.mean_position(mode=0)
    m1 = state.mean_position(mode=1)
    m2 = state.mean_position(mode=2)
    m3 = state.mean_position(mode=3)
    m4 = state.mean_position(mode=4)
    m5 = state.mean_position(mode=5)
    m6 = state.mean_position(mode=6)
    m7 = state.mean_position(mode=7)

    return tf.stack([m0,m1,m2,m3,m4,m5,m6,m7], axis=1)
