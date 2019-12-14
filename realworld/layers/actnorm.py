import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model

tfd = tfp.distributions
tfb = tfp.bijectors


class Actnorm(tfb.Bijector):
    def __init__(self,
                 channels: int,
                 validate_args=False,
                 name='actnorm',
                 log_scale_factor=1.0):
        super(Actnorm, self).__init__(
            # this bijector affect vector-wise (channel-wise) => forward_min_event_ndims=1
            forward_min_event_ndims=1,
            validate_args=validate_args,
            name=name)
        self.log_scale_factor = log_scale_factor
        self.initialized = False
        self.log_scale = tf.Variable(tf.random.normal([channels]))
        self.bias = tf.Variable(tf.random.normal([channels]))

    def setStat(self, x):
        mean = tf.math.reduce_mean(x, axis=[0, 1, 2])
        var = tf.math.reduce_mean((x - mean)**2, axis=[0, 1, 2])
        stdvar = tf.math.sqrt(var) + 1e-6
        log_scale = tf.math.log(
            1. / stdvar / self.log_scale_factor) * self.log_scale_factor
        self.bias.assign(-mean)
        self.log_scale.assign(log_scale)

    def _forward(self, x):
        if not self.initialized:
            self.setStat(x)
            self.initialized = True
        return (x + self.bias) * tf.exp(self.log_scale)

    def _inverse(self, y):
        if not self.initialized:
            self.setStat(y)
            self.initialized = True
        return y * tf.exp(-self.log_scale) - self.bias

    def _forward_log_det_jacobian(self, x):
        return tf.reduce_sum(self.log_scale)

    def _inverse_log_det_jacobian(self, y):
        return -tf.reduce_sum(self.log_scale)


def test_actnorm():
    actnorm = Actnorm(4)
    x = tf.random.normal([100, 16, 16, 4]) + 100
    y = actnorm.forward(x)
    z = actnorm.inverse(y)
    print('input: x', tf.reduce_mean(x, axis=[0, 1, 2]).numpy())
    print('output: y', tf.reduce_mean(y, axis=[0, 1, 2]).numpy())
    print('inverse: z', tf.reduce_mean(z, axis=[0, 1, 2]).numpy())
    print('log_det_jacobian: ',
          actnorm.forward_log_det_jacobian(y, event_ndims=3).numpy())
