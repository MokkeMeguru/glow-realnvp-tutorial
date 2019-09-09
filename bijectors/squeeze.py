import tensorflow_probability as tfp
import tensorflow as tf


class Squeeze(tfp.bijectors.Bijector):
    """
    assert : h % factor == 0 and w % factor == 0
    """
    def __init__(self, factor=2, validate_args=False, name='squeeze'):
        super(Squeeze, self).__init__(forward_min_event_ndims=3,
                                      validate_args=validate_args,
                                      name=name)
        self.factor = factor

    def _forward(self, x):
        [h, w, c] = x.shape[-3:]
        in_shape = [
            h // self.factor, self.factor, w // self.factor, self.factor, c
        ]
        out_shape = [h // self.factor, w // self.factor, c * self.factor**2]
        shape_prefix = tf.shape(x)[:-3]
        in_shape = tf.concat([shape_prefix, in_shape], axis=0)
        out_shape = tf.concat([shape_prefix, out_shape], axis=0)
        y = tf.reshape(x, in_shape)
        y = tf.transpose(y, [0, 1, 3, 5, 2, 4])
        y = tf.reshape(y, out_shape)
        return y

    def _inverse(self, y):
        [h, w, c] = y.shape[-3:]
        in_shape = [h, w, c // self.factor**2, self.factor, self.factor]
        out_shape = [h * self.factor, w * self.factor, c // self.factor**2]
        shape_prefix = tf.shape(y)[:-3]
        in_shape = tf.concat([shape_prefix, in_shape], axis=0)
        out_shape = tf.concat([shape_prefix, out_shape], axis=0)
        x = tf.reshape(y, in_shape)
        x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
        x = tf.reshape(x, out_shape)
        return x

    def _inverse_log_det_jacobian(self, y):
        # dx/dy=I, so log|det(dx/dy)| = 0
        ildj = tf.zeros([tf.shape(y)[0]], dtype=y.dtype)
        return ildj
