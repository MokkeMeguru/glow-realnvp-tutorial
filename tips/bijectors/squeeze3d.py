import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class Squeeze3D(tfb.Bijector):
    def __init__(
            self,
            factor=2,
            forward_min_event_ndims=0,
            inverse_min_event_ndims=0,
            validate_args=False,
            name="Squeeze",
    ):
        self._factor = factor
        super(Squeeze3D,
              self).__init__(forward_min_event_ndims=forward_min_event_ndims,
                             inverse_min_event_ndims=inverse_min_event_ndims,
                             name=name,
                             is_constant_jacobian=True)

    @property
    def factor(self):
        return self._factor

    def _forward(self, x):
        (H, W, C) = x.shape[1:]
        batch_size = tf.shape(x)[0:1]
        tmp_shape = tf.concat(
            [
                batch_size,
                (H // self.factor, self.factor, W // self.factor, self.factor,
                 C),
            ],
            axis=0,
        )
        output_shape = tf.concat(
            [
                batch_size,
                (H // self.factor, W // self.factor, C * self.factor**2)
            ],
            axis=0,
        )
        y = tf.reshape(x, tmp_shape)
        y = tf.transpose(y, [0, 1, 3, 5, 2, 4])
        y = tf.reshape(y, output_shape)
        return y

    def _inverse(self, y):
        (H, W, C) = y.shape[1:]
        batch_size = tf.shape(y)[0:1]
        tmp_shape = tf.concat([
            batch_size, (H, W, C // self.factor**2, self.factor, self.factor)
        ],
                              axis=0)
        output_shape = tf.concat([
            batch_size, (H * self.factor, W * self.factor, C // self.factor**2)
        ],
                                 axis=0)
        x = tf.reshape(y, tmp_shape)
        x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
        x = tf.reshape(x, output_shape)
        return x

    def _forward_log_det_jacobian(self, x):
        return tf.constant(0.0, dtype=x.dtype)


def test_squeeze3D():
    factor = 2
    x = tf.Variable([[[1, 2, 5, 6], [3, 4, 7, 8], [9, 10, 13, 14],
                      [11, 12, 15, 16]]])
    x = tf.expand_dims(x, axis=-1)
    squeeze3d = Squeeze3D(factor=factor)
    y = squeeze3d.forward(x)
    z = squeeze3d.inverse(y)
    print(tf.reduce_sum(x - z))

    flow = tfd.TransformedDistribution(event_shape=[16, 16, 2],
                                       distribution=tfd.Normal(loc=0.,
                                                               scale=1.),
                                       bijector=squeeze3d)
    x = tf.random.normal([64, 16, 16, 2])
    y = flow.bijector.forward(x)
    log_prob = flow.log_prob(y)
    print(x.shape, y.shape, log_prob.shape)


def main():
    test_squeeze3D()


if __name__ == '__main__':
    print('tensorflow: ', tf.__version__)
    print('tensorflow_probability: ', tfp.__version__)
    main()
