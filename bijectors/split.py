import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tensorflow.keras import layers


class _SplitLayer(layers.Layer):
    def __init__(self, filter_size):
        super(_SplitLayer, self).__init__()
        self.conv2d_1 = layers.Conv2D(filters=filter_size,
                                      kernel_size=(3, 3),
                                      padding='same')

    def call(self, x):
        return self.conv2d_1(x)


class Split(tfp.bijectors.Bijector):
    def __init__(self, input_shape, validate_args=False, name='split'):
        super(Split, self).__init__(forward_min_event_ndims=3,
                                    validate_args=validate_args,
                                    name=name)
        input_shape = (input_shape[-3], input_shape[-2], input_shape[-1])
        self.split_channels = [
            input_shape[-1] // 2, input_shape[-1] - input_shape[-1] // 2
        ]

        self.conv = _SplitLayer(input_shape[-1])

    def _forward(self, x):
        xa, xb = tf.split(x, self.split_channels, axis=-1)
        return xa

    def _inverse(self, ya):
        theta = self.conv(ya)
        mean, logs = tf.split(theta, self.split_channels, axis=-1)
        yb = tf.random.normal(mean.get_shape()) * tf.math.exp(logs) + mean
        y = tf.concat([ya, yb], axis=-1)
        return y

    def _forward_log_det_jacobian(self, x):
        xa, xb = tf.split(x, self.split_channels, axis=-1)
        theta = self.conv(xa)
        mean, logs = tf.split(theta, self.split_channels, axis=-1)
        fldj = -0.5 * (logs * 2 + tf.math.square(xb - mean) /
                       tf.math.exp(logs * 2) + tf.math.log(2 * np.pi))
        fldj = tf.math.reduce_sum(fldj, axis=[1, 2, 3])
        return fldj

    def _inverse_log_det_jacobian(self, y):
        ildj = tf.zeros([tf.shap(y)[0]], dtype=y.dtype)
        return ildj


def main():
    split = Split([None, 28, 28, 3])
    x = tf.random.normal([2, 28, 28, 3])
    y = split.forward(x)
    z = split.inverse(y)
    loss = tf.reduce_mean(z - x)
    print(loss.numpy())
