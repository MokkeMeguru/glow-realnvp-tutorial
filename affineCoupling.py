# See. https://github.com/breadbread1984/glow-flow/blob/master/AffineCoupling.py
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp


class _nn(layers.Layer):
    """
    Non-trainable params?
    These parameters are from BatchNormalization layers.
    They're updated with mean and variance, but they're not "trained with backpropagation".
    ref. https://stackoverflow.com/questions/47312219/what-does-non-trainable-params-mean/51729624
    """
    def __init__(self, hidden_filters, channels=3, activation='relu'):
        super(_nn, self).__init__()
        # self.batch_shape = batch_shape
        if activation == 'relu':
            activation_cls = layers.ReLU
        self.conv2d_1 = layers.Conv2D(filters=hidden_filters,
                                      kernel_size=(3, 3),
                                      padding='same')
        self.batch_norm_1 = layers.BatchNormalization()
        self.activation_1 = activation_cls()

        self.conv2d_2 = layers.Conv2D(filters=hidden_filters,
                                      kernel_size=(1, 1),
                                      padding='same')
        self.batch_norm_2 = layers.BatchNormalization()
        self.activation_2 = activation_cls()
        self.conv2d_out = layers.Conv2D(filters=channels,
                                        kernel_size=(3, 3),
                                        padding='same')

    def call(self, x):
        x = self.conv2d_1(x)
        x = self.batch_norm_1(x)
        x = self.activation_1(x)
        x = self.conv2d_2(x)
        x = self.batch_norm_2(x)
        x = self.activation_2(x)
        x = self.conv2d_out(x)
        return x


class AffineCoupling(tfp.bijectors.Bijector):
    def __init__(self,
                 batch_shape,
                 hidden_filters=512,
                 validate_args=False,
                 name='affine_coupling'):
        super(AffineCoupling, self).__init__(forward_min_event_ndims=3,
                                             validate_args=validate_args,
                                             name=name)
        # batch shape : [batch_size, h, w, c] ex. [None, 28, 28, 3]
        # in Glow, c = 3 is ideal?
        self.split_channels = [
            batch_shape[-1] - batch_shape[-1] // 2, batch_shape[-1] // 2
        ]
        self.nn = _nn(hidden_filters, channels=batch_shape[-1])

    def _forward(self, x):
        xa, xb = tf.split(x, self.split_channels, axis=-1)
        logs_t = self.nn(xb)
        log_scale, shifts = tf.split(logs_t, self.split_channels, axis=-1)
        scale = tf.exp(log_scale)
        ya = scale * xa + shifts
        yb = xb
        y = tf.concat([ya, yb], axis=-1)
        return y

    def _inverse(self, y):
        ya, yb = tf.split(y, self.split_channels, axis=-1)
        logs_t = self.nn(yb)
        log_scale, shifts = tf.split(logs_t, self.split_channels, axis=-1)
        scale = tf.exp(log_scale)
        xa = (ya - shifts) / scale
        xb = yb
        x = tf.concat([xa, xb], axis=-1)
        return x

    def _inverse_log_det_jacobian(self, y):
        _, yb = tf.keras.layers.Lambda(
            lambda e: tf.split(e, self.split_channels, axis=-1))(y)
        logs_t = self.nn(yb)
        log_scale, _ = tf.keras.layers.Lambda(
            lambda e: tf.split(e, self.split_channels, axis=-1))(logs_t)
        scale = tf.exp(log_scale)
        ildj = -tf.math.reduce_sum(tf.math.log(tf.abs(scale)))
        # expand batch_shape
        ildj = tf.tile([ildj], [tf.shape(y)[0]])
        return ildj


def main():
    x = tf.keras.Input([28, 28, 3])
    affine_coupling = AffineCoupling(batch_shape=[None, 28, 28, 3],
                                     hidden_filters=512)
    # print (fwd)
    # print (affine_coupling.trainable_variables)

    y = affine_coupling.forward(x)
    # z = affine_coupling.inverse(y)
    assert x.shape.as_list() == y.shape.as_list(
    ), 'Tensor Shape Error : In. {} not queal Out. {}'.format(
        x.shape, y.shape)

    tf.keras.Model(inputs=x, outputs=y).summary()
