# See. https://github.com/breadbread1984/glow-flow/blob/master/AffineCoupling.py
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_probability.python.bijectors as tfb
import tensorflow_probability.python.distributions as tfd


class AffineCoupling(tfp.bijectors.Bijector):
    def __init__(self,
                 batch_shape,
                 hidden_filters=512,
                 validate_args=False,
                 name='affine_coupling'):
        super(AffineCoupling, self).__init__(forward_min_event_ndims=3,
                                             validate_args=validate_args,
                                             name=name)
        # batch shape : [batch_size, h, w, c]
        # in Glow, c = 3 is ideal?
        batch_shape = tf.convert_to_tensor(batch_shape)
        batch_shape[-1] = batch_shape[-1] // 2
        x = tf.keras.Input(shape=batch_shape[-3:].tolist())
