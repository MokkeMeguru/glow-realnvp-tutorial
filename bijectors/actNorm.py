import tensorflow as tf
import tensorflow_probability as tfp


class ActNorm(tfp.bijectors.Bijector):
    def __init__(self, validate_args=False, name='actnorm'):
        super(ActNorm, self).__init__(forward_min_event_ndims=3,
                                      validate_args=validate_args,
                                      name=name)
        self.initialized = False

    def getStat(self, x):
        mean = tf.math.reduce_mean(x, axis=[0, 1, 2])
        variance = tf.math.reduce_mean(tf.math.square(x - mean),
                                       axis=[0, 1, 2])
        stdvar = tf.math.sqrt(variance)
        return mean, stdvar

    def _forward(self, x):
        if not self.initialized:
            mean, stdvar = self.getStat(x)
            self.initialized = True
            self.scale = tf.math.reciprocal(stdvar + 1e-6)
            self.b = mean
        return (x - self.b) * self.scale

    def _inverse(self, y):
        if not self.initialized:
            mean, stdvar = self.getStat(y)
            self.initialized = True
            self.scale = tf.math.reciprocal(stdvar + 1e-6)
            self.b = -mean
        return y / self.scale + self.b

    def _inverse_log_det_jacobian(self, y):
        shape = tf.shape(y)[-3:]
        hxw = tf.math.reduce_prod(tf.cast(shape[:-1], dtype=tf.float32))
        ildj = hxw * tf.math.reduce_sum(tf.math.log(tf.abs(self.scale)))
        # expand batch_shape
        ildj = tf.tile([ildj], [tf.shape(y)[0]])
        return ildj

def main():
    actnorm = ActNorm([None, 28, 28, 3])
    x = tf.random.normal([2, 28, 28, 3])
    y = actnorm.forward(x)
    z = actnorm.inverse(y)
    loss = tf.reduce_mean(z - x)
    print(actnorm.inverse_log_det_jacobian(y, event_ndims=3).numpy())
    print(loss.numpy())
