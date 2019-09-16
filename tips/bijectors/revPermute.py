import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class RevPermute(tfb.Bijector):
    def __init__(
            self,
            axis=[-1],
            forward_min_event_ndims=0,
            validate_args=False,
            name="RevPermute",
    ):
        super(RevPermute,
              self).__init__(forward_min_event_ndims=forward_min_event_ndims,
                             validate_args=validate_args,
                             name=name,
                             is_constant_jacobian=True)
        self._axis = axis

    @property
    def axis(self):
        return self._axis

    def _forward(self, x):
        return tf.reverse(x, self.axis)

    def _inverse(self, y):
        return tf.reverse(y, self.axis)

    def _forward_log_det_jacobian(self, x):
        return tf.constant(0.0, dtype=x.dtype)

    def _inverse_log_det_jacobian(self, y):
        return tf.constant(0.0, dtype=y.dtype)


def test_revPermute():
    target_dist = tfd.Normal(loc=0., scale=1.)
    revPermute = RevPermute()
    x = tf.keras.Input([16, 16, 4])
    y = revPermute.forward(x)
    # tf.keras.Model(x, y).summary()

    x = tf.random.normal([2, 16, 16, 4])
    y = revPermute.forward(x)
    z = revPermute.inverse(y)
    flow = tfd.TransformedDistribution(event_shape=[16, 16, 4],
                                       distribution=target_dist,
                                       bijector=revPermute)
    print(tf.reduce_mean(flow.log_prob(tf.random.normal([2, 16, 16, 4]))))
    return tf.reduce_sum(z - x)


def main():
    test_revPermute()


if __name__ == '__main__':
    print('tensorflow: ', tf.__version__)
    print('tensorflow_probability: ', tfp.__version__)
    main()
