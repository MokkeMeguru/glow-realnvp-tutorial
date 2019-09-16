import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Layer, BatchNormalization, Conv2D
from tensorflow.keras import Model

tfd = tfp.distributions
tfb = tfp.bijectors


# ref: https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/blockwise.py
# TODO: ask is the event_ndim for forward/invert_log_determinants correctly.
class Blockwise3D(tfb.Bijector):
    def __init__(self,
                 bijectors,
                 block_sizes=None,
                 validate_args=False,
                 name=None):
        if not name:
            name = "blockwise3D_of_" + "_and_".join(
                [b.name for b in bijectors])
            name = name.replace("/", "")
        super(Blockwise3D, self).__init__(
            # ???
            forward_min_event_ndims=3,
            validate_args=validate_args,
            name=name,
        )
        self._bijectors = bijectors
        self._block_sizes = block_sizes

    @property
    def bijectors(self):
        return self._bijectors

    @property
    def block_sizes(self):
        return self._block_sizes

    def _forward(self, x):
        split_x = (tf.split(x, len(self.bijectors), axis=-1)
                   if self.block_sizes is None else tf.split(
                       x, self.block_sizes, axis=-1))
        split_y = [b.forward(x_) for b, x_ in zip(self.bijectors, split_x)]
        y = tf.concat(split_y, axis=-1)
        return y

    def _inverse(self, y):
        split_y = (tf.split(y, len(self.bijectors), axis=-1)
                   if self.block_sizes is None else tf.split(
                       y, self.block_sizes, axis=-1))
        split_x = [b.inverse(y_) for b, y_ in zip(self.bijectors, split_y)]
        x = tf.concat(split_x, axis=-1)
        return x

    def _forward_log_det_jacobian(self, x):
        split_x = (tf.split(x, len(self.bijectors), axis=-1)
                   if self.block_sizes is None else tf.split(
                       x, self.block_sizes, axis=-1))
        fldjs = [
            # ???
            b.forward_log_det_jacobian(x_, event_ndims=3)
            for b, x_ in zip(self.bijectors, split_x)
        ]
        return sum(fldjs)

    def _inverse_log_det_jacobian(self, y):
        split_y = (tf.split(y, len(self.bijectors), axis=-1)
                   if self.block_sizes is None else tf.split(
                       y, self.block_sizes, axis=-1))
        ildjs = [
            b.inverse_log_det_jacobian(y_, event_ndims=3)
            for b, y_ in zip(self.bijectors, split_y)
        ]
        return sum(ildjs)


def test_blockwise3D():
    from . import RealNVP
    blockwise3D = Blockwise3D(bijectors=[
        tfb.Identity(),
        RealNVP(input_shape=[16, 16, 2], n_hidden=[256, 256]),
    ])

    x = tf.keras.Input([16, 16, 4])
    y = blockwise3D.forward(x)
    tf.keras.Model(x, y).summary()

    x = tf.random.normal([3, 16, 16, 4])
    y = blockwise3D.forward(x)
    z = blockwise3D.inverse(y)
    return tf.reduce_sum(z - x)


def main():
    print(test_blockwise3D())


if __name__ == '__main__':
    main()
